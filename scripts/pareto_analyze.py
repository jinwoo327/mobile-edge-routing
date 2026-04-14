from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.pareto import pareto_frontier, plot_frontier, write_csv


def _read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]

    for row in rows:
        for key in ("threshold", "accuracy", "mean_latency_ms", "edge_rate", "local_rate", "fallback_rate"):
            if key in row and row[key] not in {"", None}:
                row[key] = float(row[key])
        if "sample_count" in row and row["sample_count"] not in {"", None}:
            row["sample_count"] = int(float(row["sample_count"]))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute Pareto frontier from threshold sweep CSV output.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--plot-path", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_csv = Path(args.input_csv)
    rows = _read_csv_rows(input_csv)
    frontier = pareto_frontier(rows, maximize=["accuracy"], minimize=["mean_latency_ms", "edge_rate"])

    output_csv = Path(args.output_csv) if args.output_csv else input_csv.with_name(f"{input_csv.stem}-pareto.csv")
    write_csv(output_csv, frontier)

    if args.plot_path:
        try:
            plot_frontier(rows, frontier, args.plot_path)
        except RuntimeError as exc:
            print(f"plot_error={exc}")
            raise SystemExit(1)

    print(json.dumps({"pareto_csv": str(output_csv), "frontier_points": len(frontier)}, indent=2))


if __name__ == "__main__":
    main()
