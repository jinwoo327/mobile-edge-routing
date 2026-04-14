from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets.common import DatasetSample
from datasets.loaders import DatasetLoadResult, load_dataset
from evaluation.logging import append_jsonl, write_json
from evaluation.metrics import evaluate_prediction
from evaluation.summary import summarize_results, write_summary_csv
from models.backends import BackendConfig, create_edge_backend, create_local_backend
from models.ollama_client import OllamaError
from monitoring.interfaces import MockBatteryMonitor, MockNetworkMonitor
from routing.modes import ExecutionMode, execute_mode


@dataclass(slots=True)
class RunArtifacts:
    run_id: str
    jsonl_path: Path
    summary_csv_path: Path
    metadata_path: Path
    sample_count: int
    summary_rows: int

def _load_samples(
    dataset_name: str,
    dataset_source: str,
    limit: int | None,
    data_path: str | None,
) -> DatasetLoadResult:
    if dataset_name != "all":
        return load_dataset(dataset_name, path=data_path, limit=limit, dataset_source=dataset_source)

    combined_samples: list[DatasetSample] = []
    resolved_paths: list[str] = []
    warnings: list[str] = []
    available_count = 0

    root_path = Path(data_path) if data_path else None
    for name in ("mmlu", "gsm8k"):
        child_path = None
        if root_path is not None and root_path.is_dir():
            child_path = root_path / name
        elif root_path is not None:
            child_path = root_path

        result = load_dataset(name, path=child_path, limit=None, dataset_source=dataset_source)
        combined_samples.extend(result.samples)
        resolved_paths.extend(result.resolved_paths)
        available_count += result.available_count
        if result.warning:
            warnings.append(result.warning)

    selected_samples = combined_samples[:limit] if limit is not None else combined_samples
    warning = None
    if limit is not None and limit > available_count:
        warning = (
            f"Requested limit {limit} for all datasets ({dataset_source}) but only "
            f"{available_count} samples were available."
        )
    elif warnings:
        warning = "; ".join(warnings)

    return DatasetLoadResult(
        dataset_name=dataset_name,
        dataset_source=dataset_source,
        samples=selected_samples,
        available_count=available_count,
        requested_limit=limit,
        resolved_paths=resolved_paths,
        warning=warning,
    )


def run_experiment(
    dataset_name: str,
    mode: ExecutionMode,
    dataset_source: str = "fixture",
    limit: int | None = None,
    confidence_threshold: float = 0.6,
    block_sizes: list[int] | None = None,
    data_path: str | None = None,
    log_dir: str | Path = "logs",
    local_backend_type: str = "mock_local_backend",
    local_base_url: str = "http://localhost:11434",
    local_model_name: str = "gemma3:1b",
    local_timeout: float = 60.0,
    local_quantization: str | None = None,
    local_confidence_estimator: str = "router_v1",
    edge_backend_type: str = "mock_edge_backend",
    edge_base_url: str = "http://localhost:11434",
    edge_model_name: str = "gemma3:4b",
    edge_timeout: float = 60.0,
    edge_quantization: str | None = None,
) -> RunArtifacts:
    block_sizes = block_sizes or [20, 50, 100]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_path = Path(log_dir)
    jsonl_path = log_path / f"run-{run_id}.jsonl"
    summary_csv_path = log_path / f"summary-{run_id}.csv"
    metadata_path = log_path / f"run-{run_id}-metadata.json"

    local_backend_config = BackendConfig(
        backend_type=local_backend_type,
        base_url=local_base_url,
        model_name=local_model_name,
        timeout=local_timeout,
        quantization=local_quantization,
        confidence_estimator=local_confidence_estimator,
    )
    edge_backend_config = BackendConfig(
        backend_type=edge_backend_type,
        model_name=edge_model_name,
        quantization=edge_quantization,
        base_url=edge_base_url,
        timeout=edge_timeout,
    )

    local_backend = create_local_backend(local_backend_config)
    edge_backend = create_edge_backend(edge_backend_config)
    battery_monitor = MockBatteryMonitor()
    network_monitor = MockNetworkMonitor()

    load_result = _load_samples(
        dataset_name=dataset_name,
        dataset_source=dataset_source,
        limit=limit,
        data_path=data_path,
    )
    samples = load_result.samples
    results: list[dict] = []
    local_health: dict[str, Any] | None = None
    edge_health: dict[str, Any] | None = None

    if mode in {"local_only", "hybrid_only"} and hasattr(local_backend, "ensure_ready"):
        try:
            local_health = local_backend.ensure_ready()
        except OllamaError:
            raise

    if mode in {"edge_only", "hybrid_only"} and hasattr(edge_backend, "ensure_ready"):
        try:
            edge_health = edge_backend.ensure_ready()
        except OllamaError:
            raise

    run_metadata = {
        "run_id": run_id,
        "dataset": dataset_name,
        "dataset_source": dataset_source,
        "mode": mode,
        "limit": limit,
        "confidence_threshold": confidence_threshold,
        "block_sizes": block_sizes,
        "data_path": data_path,
        "resolved_data_paths": load_result.resolved_paths,
        "dataset_available_count": load_result.available_count,
        "dataset_warning": load_result.warning,
        "local_backend": {
            "backend_type": local_backend_type,
            "base_url": local_base_url,
            "model_name": local_model_name,
            "timeout": local_timeout,
            "quantization": local_quantization,
            "confidence_estimator": local_confidence_estimator,
        },
        "edge_backend": {
            "backend_type": edge_backend_type,
            "base_url": edge_base_url,
            "model_name": edge_model_name,
            "timeout": edge_timeout,
            "quantization": edge_quantization,
        },
        "local_health": local_health,
        "edge_health": edge_health,
    }
    write_json(metadata_path, run_metadata)

    if load_result.warning:
        print(f"warning={load_result.warning}")

    for sample in samples:
        battery_before = asdict(battery_monitor.snapshot())
        network_before = asdict(network_monitor.snapshot())

        decision = execute_mode(
            sample=sample,
            mode=mode,
            local_backend=local_backend,
            edge_backend=edge_backend,
            confidence_threshold=confidence_threshold,
        )

        battery_after = asdict(battery_monitor.snapshot())
        network_after = asdict(network_monitor.snapshot())
        is_correct = evaluate_prediction(sample, decision.final_result.prediction)
        backend_role = "local" if decision.local_result and decision.backend_used == decision.local_result.backend_name else "edge"

        record = {
            "run_id": run_id,
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "dataset_source": dataset_source,
            "mode": decision.mode,
            "question": sample.question,
            "gold_answer": sample.answer,
            "prediction": decision.final_result.prediction,
            "is_correct": is_correct,
            "confidence": decision.final_result.confidence,
            "latency_ms": decision.final_result.latency_ms,
            "backend_used": decision.backend_used,
            "backend_role": backend_role,
            "backend_metadata": decision.final_result.metadata,
            "confidence_source": decision.final_result.metadata.get("confidence_source"),
            "fell_back_to_edge": decision.fell_back_to_edge,
            "local_confidence": decision.local_result.confidence if decision.local_result else None,
            "local_confidence_source": (
                decision.local_result.metadata.get("confidence_source") if decision.local_result else None
            ),
            "local_backend_type": local_backend_type,
            "local_model_name": local_model_name,
            "local_quantization": local_quantization,
            "local_confidence_estimator": local_confidence_estimator,
            "edge_backend_type": edge_backend_type,
            "edge_model_name": edge_model_name,
            "edge_quantization": edge_quantization,
            "run_metadata": run_metadata,
            "battery_before": battery_before,
            "battery_after": battery_after,
            "network_before": network_before,
            "network_after": network_after,
        }
        append_jsonl(jsonl_path, record)
        results.append(record)

    summary_rows = summarize_results(results, block_sizes=block_sizes)
    write_summary_csv(summary_csv_path, summary_rows)

    return RunArtifacts(
        run_id=run_id,
        jsonl_path=jsonl_path,
        summary_csv_path=summary_csv_path,
        metadata_path=metadata_path,
        sample_count=len(results),
        summary_rows=len(summary_rows),
    )
