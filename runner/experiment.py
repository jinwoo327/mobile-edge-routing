from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import traceback
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
    error_path: Path | None
    error_log_path: Path | None
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
    skip_failed_samples: bool = False,
    prompt_profile: str = "benchmark_v2",
    block_sizes: list[int] | None = None,
    data_path: str | None = None,
    log_dir: str | Path = "logs",
    local_backend_type: str = "mock_local_backend",
    local_base_url: str = "http://localhost:11434",
    local_model_name: str = "gemma3:1b",
    local_timeout: float = 60.0,
    local_quantization: str | None = None,
    local_temperature: float | None = 0.0,
    local_confidence_estimator: str = "router_v1",
    edge_backend_type: str = "mock_edge_backend",
    edge_base_url: str = "http://localhost:11434",
    edge_model_name: str = "gemma3:4b",
    edge_timeout: float = 60.0,
    edge_quantization: str | None = None,
    edge_temperature: float | None = 0.0,
) -> RunArtifacts:
    block_sizes = block_sizes or [20, 50, 100]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    log_path = Path(log_dir)
    jsonl_path = log_path / f"run-{run_id}.jsonl"
    summary_csv_path = log_path / f"summary-{run_id}.csv"
    metadata_path = log_path / f"run-{run_id}-metadata.json"
    error_path = log_path / f"run-{run_id}-error.json"
    error_log_path = log_path / f"run-{run_id}-errors.jsonl"

    local_backend_config = BackendConfig(
        backend_type=local_backend_type,
        base_url=local_base_url,
        model_name=local_model_name,
        timeout=local_timeout,
        quantization=local_quantization,
        confidence_estimator=local_confidence_estimator,
        temperature=local_temperature,
        prompt_profile=prompt_profile,
    )
    edge_backend_config = BackendConfig(
        backend_type=edge_backend_type,
        model_name=edge_model_name,
        quantization=edge_quantization,
        base_url=edge_base_url,
        timeout=edge_timeout,
        temperature=edge_temperature,
        prompt_profile=prompt_profile,
    )

    local_backend = create_local_backend(local_backend_config)
    edge_backend = create_edge_backend(edge_backend_config)
    battery_monitor = MockBatteryMonitor()
    network_monitor = MockNetworkMonitor()

    load_result = _load_samples(
        dataset_name=dataset_name,
        dataset_source=dataset_source,
        limit=None if skip_failed_samples else limit,
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
        "skip_failed_samples": skip_failed_samples,
        "prompt_profile": prompt_profile,
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
            "temperature": local_temperature,
            "confidence_estimator": local_confidence_estimator,
            "prompt_profile": prompt_profile,
        },
        "edge_backend": {
            "backend_type": edge_backend_type,
            "base_url": edge_base_url,
            "model_name": edge_model_name,
            "timeout": edge_timeout,
            "quantization": edge_quantization,
            "temperature": edge_temperature,
            "prompt_profile": prompt_profile,
        },
        "local_health": local_health,
        "edge_health": edge_health,
    }
    write_json(metadata_path, run_metadata)

    if load_result.warning:
        print(f"warning={load_result.warning}")

    for sample_index, sample in enumerate(samples, start=1):
        battery_before = asdict(battery_monitor.snapshot())
        network_before = asdict(network_monitor.snapshot())

        try:
            decision = execute_mode(
                sample=sample,
                mode=mode,
                local_backend=local_backend,
                edge_backend=edge_backend,
                confidence_threshold=confidence_threshold,
            )
        except Exception as exc:
            error_payload = {
                "run_id": run_id,
                "dataset": dataset_name,
                "dataset_source": dataset_source,
                "mode": mode,
                "skip_failed_samples": skip_failed_samples,
                "prompt_profile": prompt_profile,
                "failed_sample_index": sample_index,
                "failed_sample_id": sample.sample_id,
                "completed_sample_count": len(results),
                "requested_limit": limit,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            append_jsonl(error_log_path, error_payload)
            if skip_failed_samples:
                print(f"warning=Skipping failed sample {sample_index} ({sample.sample_id}) due to {type(exc).__name__}")
                continue

            write_json(error_path, error_payload)
            if results:
                summary_rows = summarize_results(results, block_sizes=block_sizes)
                write_summary_csv(summary_csv_path, summary_rows)
            print(f"error=Run failed at sample {sample_index} ({sample.sample_id})")
            print(f"error_json={error_path}")
            raise

        battery_after = asdict(battery_monitor.snapshot())
        network_after = asdict(network_monitor.snapshot())
        is_correct = evaluate_prediction(sample, decision.final_result.prediction)
        backend_role = "local" if decision.local_result and decision.backend_used == decision.local_result.backend_name else "edge"
        local_latency_ms = decision.local_result.latency_ms if decision.local_result else None
        edge_latency_ms = decision.edge_result.latency_ms if decision.edge_result else None
        final_backend_latency_ms = decision.final_result.latency_ms
        end_to_end_latency_ms = _compute_end_to_end_latency_ms(decision)

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
            # latency_ms is the comparison-facing end-to-end latency for the selected execution mode.
            "latency_ms": end_to_end_latency_ms,
            "end_to_end_latency_ms": end_to_end_latency_ms,
            "final_backend_latency_ms": final_backend_latency_ms,
            "local_latency_ms": local_latency_ms,
            "edge_latency_ms": edge_latency_ms,
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
            "local_temperature": local_temperature,
            "local_confidence_estimator": local_confidence_estimator,
            "edge_backend_type": edge_backend_type,
            "edge_model_name": edge_model_name,
            "edge_quantization": edge_quantization,
            "edge_temperature": edge_temperature,
            "prompt_profile": prompt_profile,
            "run_metadata": run_metadata,
            "battery_before": battery_before,
            "battery_after": battery_after,
            "network_before": network_before,
            "network_after": network_after,
        }
        append_jsonl(jsonl_path, record)
        results.append(record)
        if limit is not None and len(results) >= limit:
            break

    if skip_failed_samples and limit is not None and len(results) < limit:
        print(
            "warning="
            f"Requested {limit} successful samples for {dataset_name} ({dataset_source}) but only "
            f"{len(results)} completed successfully before the dataset was exhausted."
        )

    summary_rows = summarize_results(results, block_sizes=block_sizes)
    write_summary_csv(summary_csv_path, summary_rows)

    return RunArtifacts(
        run_id=run_id,
        jsonl_path=jsonl_path,
        summary_csv_path=summary_csv_path,
        metadata_path=metadata_path,
        error_path=error_path if error_path.exists() else None,
        error_log_path=error_log_path if error_log_path.exists() else None,
        sample_count=len(results),
        summary_rows=len(summary_rows),
    )


def _compute_end_to_end_latency_ms(decision: Any) -> float:
    latencies = []
    if decision.local_result is not None:
        latencies.append(decision.local_result.latency_ms)
    if decision.edge_result is not None:
        latencies.append(decision.edge_result.latency_ms)
    if not latencies:
        latencies.append(decision.final_result.latency_ms)
    return round(sum(latencies), 2)
