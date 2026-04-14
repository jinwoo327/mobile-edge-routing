from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.compare import compare_runs, plot_quality_latency, write_comparison_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare completed run JSONL files and summarize quality-latency tradeoffs.")
    parser.add_argument("--run-ids", nargs="*", default=None, help="Run ids such as 20260414-191938.")
    parser.add_argument("--jsonl-paths", nargs="*", default=None, help="Explicit paths to run JSONL files.")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--plot-path", default=None)
    parser.add_argument(
        "--exclude-first-n-latency",
        type=int,
        default=0,
        help="Exclude the first N samples from latency averaging to reduce model warm-up effects.",
    )
    return parser


def _resolve_paths(run_ids: list[str] | None, jsonl_paths: list[str] | None, logs_dir: str) -> list[Path]:
    resolved: list[Path] = []
    if run_ids:
        log_root = Path(logs_dir)
        resolved.extend(log_root / f"run-{run_id}.jsonl" for run_id in run_ids)
    if jsonl_paths:
        resolved.extend(Path(path) for path in jsonl_paths)
    if not resolved:
        raise SystemExit("Provide at least one run via --run-ids or --jsonl-paths.")
    return resolved


def main() -> None:
    args = build_parser().parse_args()
    jsonl_paths = _resolve_paths(args.run_ids, args.jsonl_paths, args.logs_dir)
    rows = compare_runs(jsonl_paths, exclude_first_n_latency=args.exclude_first_n_latency)

    output_csv = Path(args.output_csv) if args.output_csv else Path(args.logs_dir) / "comparison.csv"
    write_comparison_csv(output_csv, rows)

    if args.plot_path:
        try:
            x_key = "mean_latency_ms_excluding_warmup" if args.exclude_first_n_latency > 0 else "mean_latency_ms"
            plot_quality_latency(rows, args.plot_path, x_key=x_key)
        except RuntimeError as exc:
            print(f"plot_error={exc}")
            raise SystemExit(1)

    print(
        json.dumps(
            {
                "comparison_csv": str(output_csv),
                "run_count": len(rows),
                "exclude_first_n_latency": args.exclude_first_n_latency,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
