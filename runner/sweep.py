from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from runner.experiment import RunArtifacts, run_experiment


@dataclass(slots=True)
class SweepArtifacts:
    sweep_id: str
    csv_path: Path
    rows: list[dict[str, Any]]


def summarize_jsonl_run(path: str | Path, exclude_first_n_latency: int = 0) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {path}")

    edge_rate = sum(row["backend_role"] == "edge" for row in rows) / len(rows)
    local_accept_rows = [row for row in rows if row["backend_role"] == "local"]
    fallback_rows = [row for row in rows if row["fell_back_to_edge"]]
    accuracies = [1.0 if row["is_correct"] else 0.0 for row in rows]
    local_confidences = [row["local_confidence"] for row in rows if row["local_confidence"] is not None]
    latency_metric = "end_to_end_latency_ms" if any("end_to_end_latency_ms" in row for row in rows) else "latency_ms"
    all_latencies = [_get_latency_value(row) for row in rows]
    latency_rows = rows[exclude_first_n_latency:] if exclude_first_n_latency > 0 else rows
    latency_sample_count = len(latency_rows)
    excluded_latencies = [_get_latency_value(row) for row in latency_rows]
    mean_latency_excluding_warmup = round(mean(excluded_latencies), 2) if excluded_latencies else None
    median_latency_excluding_warmup = round(median(excluded_latencies), 2) if excluded_latencies else None

    return {
        "sample_count": len(rows),
        "accuracy": round(mean(accuracies), 4),
        "mean_latency_ms": round(mean(all_latencies), 2),
        "median_latency_ms": round(median(all_latencies), 2),
        "p25_latency_ms": round(_percentile(all_latencies, 0.25), 2),
        "p75_latency_ms": round(_percentile(all_latencies, 0.75), 2),
        "latency_metric": latency_metric,
        "latency_exclude_first_n": exclude_first_n_latency,
        "latency_sample_count": latency_sample_count,
        "mean_latency_ms_excluding_warmup": mean_latency_excluding_warmup,
        "median_latency_ms_excluding_warmup": median_latency_excluding_warmup,
        "p25_latency_ms_excluding_warmup": (
            round(_percentile(excluded_latencies, 0.25), 2) if excluded_latencies else None
        ),
        "p75_latency_ms_excluding_warmup": (
            round(_percentile(excluded_latencies, 0.75), 2) if excluded_latencies else None
        ),
        "edge_rate": round(edge_rate, 4),
        "local_rate": round(1.0 - edge_rate, 4),
        "fallback_rate": round(sum(row["fell_back_to_edge"] for row in rows) / len(rows), 4),
        "mean_local_confidence": round(mean(local_confidences), 4) if local_confidences else None,
        "local_accept_accuracy": (
            round(mean(1.0 if row["is_correct"] else 0.0 for row in local_accept_rows), 4)
            if local_accept_rows
            else None
        ),
        "edge_fallback_accuracy": (
            round(mean(1.0 if row["is_correct"] else 0.0 for row in fallback_rows), 4)
            if fallback_rows
            else None
        ),
        "local_confidence_source": rows[0].get("local_confidence_source"),
        "local_backend_type": rows[0].get("local_backend_type"),
        "local_model_name": rows[0].get("local_model_name"),
        "local_quantization": rows[0].get("local_quantization"),
        "local_temperature": rows[0].get("local_temperature"),
        "edge_backend_type": rows[0].get("edge_backend_type"),
        "edge_model_name": rows[0].get("edge_model_name"),
        "edge_quantization": rows[0].get("edge_quantization"),
        "edge_temperature": rows[0].get("edge_temperature"),
        "prompt_profile": rows[0].get("prompt_profile"),
        "dataset": rows[0].get("dataset"),
        "dataset_source": rows[0].get("dataset_source"),
        "mode": rows[0].get("mode"),
    }


def _get_latency_value(row: dict[str, Any]) -> float:
    if row.get("end_to_end_latency_ms") is not None:
        return float(row["end_to_end_latency_ms"])
    return float(row["latency_ms"])


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Percentile requires at least one value")
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)

    if lower_index == upper_index:
        return ordered[lower_index]

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def write_sweep_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_threshold_sweep(
    thresholds: list[float],
    output_dir: str | Path = "logs/sweeps",
    **experiment_kwargs: Any,
) -> SweepArtifacts:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sweep_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    rows: list[dict[str, Any]] = []

    for threshold in thresholds:
        artifacts: RunArtifacts = run_experiment(confidence_threshold=threshold, **experiment_kwargs)
        summary = summarize_jsonl_run(artifacts.jsonl_path)
        summary.update(
            {
                "sweep_id": sweep_id,
                "threshold": round(threshold, 4),
                "run_id": artifacts.run_id,
                "jsonl_path": str(artifacts.jsonl_path),
                "summary_csv_path": str(artifacts.summary_csv_path),
                "metadata_path": str(artifacts.metadata_path),
            }
        )
        rows.append(summary)

    csv_path = output_path / f"sweep-{sweep_id}.csv"
    write_sweep_csv(csv_path, rows)
    return SweepArtifacts(sweep_id=sweep_id, csv_path=csv_path, rows=rows)
