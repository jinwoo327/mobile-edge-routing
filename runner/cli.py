from __future__ import annotations

import argparse

from runner.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-turn mobile-edge routing experiments.")
    parser.add_argument("--dataset", choices=["mmlu", "gsm8k", "all"], default="all")
    parser.add_argument("--data-path", default=None, help="Optional path to a dataset JSONL file.")
    parser.add_argument("--mode", choices=["local_only", "edge_only", "hybrid_only"], default="hybrid_only")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to run.")
    parser.add_argument("--confidence-threshold", type=float, default=0.6)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--log-dir", default="logs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = run_experiment(
        dataset_name=args.dataset,
        mode=args.mode,
        limit=args.limit,
        confidence_threshold=args.confidence_threshold,
        block_sizes=args.block_sizes,
        data_path=args.data_path,
        log_dir=args.log_dir,
    )
    print(f"run_id={artifacts.run_id}")
    print(f"samples={artifacts.sample_count}")
    print(f"jsonl={artifacts.jsonl_path}")
    print(f"summary_csv={artifacts.summary_csv_path}")


if __name__ == "__main__":
    main()
