from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from datasets.common import DatasetSample
from datasets.loaders import load_dataset
from evaluation.logging import append_jsonl
from evaluation.metrics import evaluate_prediction
from evaluation.summary import summarize_results, write_summary_csv
from models.backends import MockEdgeBackend, MockLocalBackend
from monitoring.interfaces import MockBatteryMonitor, MockNetworkMonitor
from routing.modes import ExecutionMode, execute_mode


@dataclass(slots=True)
class RunArtifacts:
    run_id: str
    jsonl_path: Path
    summary_csv_path: Path
    sample_count: int
    summary_rows: int


def _load_samples(dataset_name: str, limit: int | None, data_path: str | None) -> list[DatasetSample]:
    if dataset_name == "all":
        samples = load_dataset("mmlu")
        samples.extend(load_dataset("gsm8k"))
        return samples[:limit] if limit else samples
    return load_dataset(dataset_name, path=data_path, limit=limit)


def run_experiment(
    dataset_name: str,
    mode: ExecutionMode,
    limit: int | None = None,
    confidence_threshold: float = 0.6,
    block_sizes: list[int] | None = None,
    data_path: str | None = None,
    log_dir: str | Path = "logs",
) -> RunArtifacts:
    block_sizes = block_sizes or [20, 50, 100]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = Path(log_dir)
    jsonl_path = log_path / f"run-{run_id}.jsonl"
    summary_csv_path = log_path / f"summary-{run_id}.csv"

    local_backend = MockLocalBackend()
    edge_backend = MockEdgeBackend()
    battery_monitor = MockBatteryMonitor()
    network_monitor = MockNetworkMonitor()

    samples = _load_samples(dataset_name=dataset_name, limit=limit, data_path=data_path)
    results: list[dict] = []

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

        record = {
            "run_id": run_id,
            "sample_id": sample.sample_id,
            "dataset": sample.dataset,
            "mode": decision.mode,
            "question": sample.question,
            "gold_answer": sample.answer,
            "prediction": decision.final_result.prediction,
            "is_correct": is_correct,
            "confidence": decision.final_result.confidence,
            "latency_ms": decision.final_result.latency_ms,
            "backend_used": decision.backend_used,
            "fell_back_to_edge": decision.fell_back_to_edge,
            "local_confidence": decision.local_result.confidence if decision.local_result else None,
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
        sample_count=len(results),
        summary_rows=len(summary_rows),
    )
