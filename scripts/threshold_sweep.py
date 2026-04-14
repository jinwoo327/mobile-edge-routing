from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.sweep import run_threshold_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a threshold sweep for hybrid routing.")
    parser.add_argument("--dataset", choices=["gsm8k", "mmlu", "all"], required=True)
    parser.add_argument("--dataset-source", choices=["fixture", "real"], default="real")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.2, 0.4, 0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--output-dir", default="logs/sweeps")
    parser.add_argument("--local-backend-type", choices=["mock_local_backend", "real_local_backend"], default="real_local_backend")
    parser.add_argument("--local-base-url", default="http://localhost:11434")
    parser.add_argument("--local-model-name", default="gemma3:1b")
    parser.add_argument("--local-timeout", type=float, default=60.0)
    parser.add_argument("--local-quantization", default=None)
    parser.add_argument("--local-confidence-estimator", choices=["format_heuristic", "router_v1"], default="router_v1")
    parser.add_argument("--edge-backend-type", choices=["mock_edge_backend", "ollama_edge_backend"], default="ollama_edge_backend")
    parser.add_argument("--edge-base-url", default="http://localhost:11434")
    parser.add_argument("--edge-model-name", default="gemma3:4b")
    parser.add_argument("--edge-timeout", type=float, default=60.0)
    parser.add_argument("--edge-quantization", default=None)
    parser.add_argument("--log-dir", default="logs")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifacts = run_threshold_sweep(
        thresholds=args.thresholds,
        output_dir=args.output_dir,
        dataset_name=args.dataset,
        dataset_source=args.dataset_source,
        data_path=args.data_path,
        mode="hybrid_only",
        limit=args.limit,
        log_dir=args.log_dir,
        local_backend_type=args.local_backend_type,
        local_base_url=args.local_base_url,
        local_model_name=args.local_model_name,
        local_timeout=args.local_timeout,
        local_quantization=args.local_quantization,
        local_confidence_estimator=args.local_confidence_estimator,
        edge_backend_type=args.edge_backend_type,
        edge_base_url=args.edge_base_url,
        edge_model_name=args.edge_model_name,
        edge_timeout=args.edge_timeout,
        edge_quantization=args.edge_quantization,
    )
    print(json.dumps({"sweep_id": artifacts.sweep_id, "csv_path": str(artifacts.csv_path)}, indent=2))


if __name__ == "__main__":
    main()
