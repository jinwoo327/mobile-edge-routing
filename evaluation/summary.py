from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def summarize_results(results: list[dict[str, Any]], block_sizes: list[int]) -> list[dict[str, Any]]:
    if not results:
        return []

    summaries: list[dict[str, Any]] = []
    grouped_results: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results:
        grouped_results.setdefault((result["dataset"], result["mode"]), []).append(result)

    for (dataset, mode), group in grouped_results.items():
        for block_size in block_sizes:
            for start in range(0, len(group), block_size):
                block = group[start : start + block_size]
                if not block:
                    continue

                summaries.append(
                    {
                        "dataset": dataset,
                        "mode": mode,
                        "block_size": block_size,
                        "block_index": (start // block_size) + 1,
                        "sample_count": len(block),
                        "accuracy": round(sum(item["is_correct"] for item in block) / len(block), 4),
                        "fallback_rate": round(sum(item["fell_back_to_edge"] for item in block) / len(block), 4),
                        "mean_latency_ms": round(sum(item["latency_ms"] for item in block) / len(block), 2),
                        "local_share": round(sum(item["backend_used"].startswith("mock_local") for item in block) / len(block), 4),
                        "edge_share": round(sum(item["backend_used"].startswith("mock_edge") for item in block) / len(block), 4),
                    }
                )
    return summaries


def write_summary_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "mode",
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
