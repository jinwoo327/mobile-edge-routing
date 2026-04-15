from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def build_presentation_rows(compare_csv_path: str | Path) -> list[dict[str, Any]]:
    path = Path(compare_csv_path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    presentation_rows: list[dict[str, Any]] = []
    for row in rows:
        presentation_rows.append(
            {
                "mode": row["mode"],
                "dataset": row["dataset"],
                "dataset_source": row["dataset_source"],
                "sample_count": _parse_int(row["sample_count"]),
                "accuracy": _parse_float(row["accuracy"]),
                "median_latency_ms": _parse_float(row["median_latency_ms"]),
                "median_latency_ms_excluding_warmup": _parse_float(row["median_latency_ms_excluding_warmup"]),
                "latency_iqr_ms_excluding_warmup": _format_iqr(
                    row["p25_latency_ms_excluding_warmup"],
                    row["p75_latency_ms_excluding_warmup"],
                ),
                "edge_rate": _parse_float(row["edge_rate"]),
                "fallback_rate": _parse_float(row["fallback_rate"]),
                "confidence_threshold": _parse_nullable_float(row.get("confidence_threshold")),
                "local_confidence_estimator": row.get("local_confidence_estimator") or None,
                "prompt_profile": row.get("prompt_profile") or None,
                "local_temperature": _parse_nullable_float(row.get("local_temperature")),
                "edge_temperature": _parse_nullable_float(row.get("edge_temperature")),
                "run_id": row["run_id"],
            }
        )

    return presentation_rows


def write_presentation_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
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


def write_presentation_json(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def _format_iqr(p25: str, p75: str) -> str:
    return f"{_parse_float(p25):.2f}-{_parse_float(p75):.2f}"


def _parse_float(value: str) -> float:
    return round(float(value), 4)


def _parse_nullable_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return round(float(value), 4)


def _parse_int(value: str) -> int:
    return int(value)
