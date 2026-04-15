from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from runner.sweep import summarize_jsonl_run


MODE_ORDER = {"local_only": 0, "hybrid_only": 1, "edge_only": 2}


def summarize_run_with_metadata(jsonl_path: str | Path, exclude_first_n_latency: int = 0) -> dict[str, Any]:
    path = Path(jsonl_path)
    summary = summarize_jsonl_run(path, exclude_first_n_latency=exclude_first_n_latency)

    run_id = _extract_run_id(path)
    metadata_path = path.with_name(f"run-{run_id}-metadata.json")
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    summary.update(
        {
            "run_id": run_id,
            "jsonl_path": str(path),
            "metadata_path": str(metadata_path) if metadata_path.exists() else None,
            "confidence_threshold": metadata.get("confidence_threshold"),
            "local_confidence_estimator": (
                metadata.get("local_backend", {}).get("confidence_estimator")
                if metadata
                else summary.get("local_confidence_source")
            ),
        }
    )
    return summary


def compare_runs(jsonl_paths: list[str | Path], exclude_first_n_latency: int = 0) -> list[dict[str, Any]]:
    rows = [summarize_run_with_metadata(path, exclude_first_n_latency=exclude_first_n_latency) for path in jsonl_paths]
    return sorted(rows, key=lambda row: (MODE_ORDER.get(str(row.get("mode")), 99), str(row.get("run_id"))))


def write_comparison_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
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


def plot_quality_latency(
    rows: list[dict[str, Any]],
    output_path: str | Path,
    x_key: str = "median_latency_ms",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for comparison plotting. Install it with: pip install matplotlib"
        ) from exc

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    colors = {
        "local_only": "#2E7D32",
        "hybrid_only": "#EF6C00",
        "edge_only": "#1565C0",
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    for row in rows:
        mode = str(row.get("mode"))
        ax.scatter(
            row[x_key],
            row["accuracy"],
            s=90,
            color=colors.get(mode, "#555555"),
            alpha=0.9,
        )
        ax.annotate(
            mode,
            (row[x_key], row["accuracy"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )

    x_labels = {
        "mean_latency_ms": "Mean Latency (ms)",
        "median_latency_ms": "Median Latency (ms)",
        "mean_latency_ms_excluding_warmup": "Mean Latency Excluding Warm-up (ms)",
        "median_latency_ms_excluding_warmup": "Median Latency Excluding Warm-up (ms)",
    }
    x_label = x_labels.get(x_key, x_key)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Accuracy")
    ax.set_title("Quality-Latency Comparison Across Modes")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _extract_run_id(path: Path) -> str:
    stem = path.stem
    prefix = "run-"
    if not stem.startswith(prefix):
        raise ValueError(f"Expected a run JSONL path like 'run-<id>.jsonl', got {path}")
    return stem[len(prefix) :]
