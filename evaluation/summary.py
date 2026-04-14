from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def summarize_results(results: list[dict[str, Any]], block_sizes: list[int]) -> list[dict[str, Any]]:
    if not results:
        return []

    summaries: list[dict[str, Any]] = []
    grouped_results: dict[tuple[str, str, str | None], list[dict[str, Any]]] = {}
    for result in results:
        grouped_results.setdefault((result["dataset"], result["mode"], result.get("dataset_source")), []).append(result)

    for (dataset, mode, dataset_source), group in grouped_results.items():
        for block_size in block_sizes:
            for start in range(0, len(group), block_size):
                block = group[start : start + block_size]
                if not block:
                    continue

                summaries.append(
                    {
                        "dataset": dataset,
                        "dataset_source": dataset_source,
                        "mode": mode,
                        "local_backend_type": block[0].get("local_backend_type"),
                        "local_model_name": block[0].get("local_model_name"),
                        "local_quantization": block[0].get("local_quantization"),
                        "local_confidence_estimator": block[0].get("local_confidence_estimator"),
                        "edge_backend_type": block[0].get("edge_backend_type"),
                        "edge_model_name": block[0].get("edge_model_name"),
                        "edge_quantization": block[0].get("edge_quantization"),
                        "block_size": block_size,
                        "block_index": (start // block_size) + 1,
                        "sample_count": len(block),
                        "accuracy": round(sum(item["is_correct"] for item in block) / len(block), 4),
                        "fallback_rate": round(sum(item["fell_back_to_edge"] for item in block) / len(block), 4),
                        "mean_latency_ms": round(sum(item["latency_ms"] for item in block) / len(block), 2),
                        "local_share": round(sum(item["backend_role"] == "local" for item in block) / len(block), 4),
                        "edge_share": round(sum(item["backend_role"] == "edge" for item in block) / len(block), 4),
                    }
                )
    return summaries


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "dataset_source",
        "mode",
        "local_backend_type",
        "local_model_name",
        "local_quantization",
        "local_confidence_estimator",
        "edge_backend_type",
        "edge_model_name",
        "edge_quantization",
        "block_size",
        "block_index",
        "sample_count",
        "accuracy",
        "fallback_rate",
        "mean_latency_ms",
        "local_share",
        "edge_share",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
