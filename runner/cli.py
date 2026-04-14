from __future__ import annotations

import argparse
import json

from models.backends import BackendConfig, create_edge_backend, create_local_backend
from models.ollama_client import OllamaError
from runner.config import apply_cli_overrides, load_run_config
from runner.experiment import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run single-turn mobile-edge routing experiments.")
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    parser.add_argument("--dataset", choices=["mmlu", "gsm8k", "all"], default=None)
    parser.add_argument("--dataset-source", choices=["fixture", "real"], default=None)
    parser.add_argument("--data-path", default=None, help="Optional path to a dataset JSONL file.")
    parser.add_argument("--mode", choices=["local_only", "edge_only", "hybrid_only"], default=None)
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of samples to run.")
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--local-backend-type", choices=["mock_local_backend", "real_local_backend"], default=None)
    parser.add_argument("--local-base-url", default=None)
    parser.add_argument("--local-model-name", default=None)
    parser.add_argument("--local-timeout", type=float, default=None)
    parser.add_argument("--local-quantization", default=None)
    parser.add_argument("--local-confidence-estimator", choices=["format_heuristic", "router_v1"], default=None)
    parser.add_argument("--edge-backend-type", choices=["mock_edge_backend", "ollama_edge_backend"], default=None)
    parser.add_argument("--edge-base-url", default=None)
    parser.add_argument("--edge-model-name", default=None)
    parser.add_argument("--edge-timeout", type=float, default=None)
    parser.add_argument("--edge-quantization", default=None)
    parser.add_argument(
        "--check-local-health",
        action="store_true",
        help="Validate that the configured local backend is reachable, then exit.",
    )
    parser.add_argument(
        "--check-edge-health",
        action="store_true",
        help="Validate that the configured edge backend is reachable, then exit.",
    )
    return parser


def _check_backend_health(backend, backend_type: str, model_name: str) -> int:
    if not hasattr(backend, "health_check"):
        print(json.dumps({"backend_type": backend_type, "reachable": True}, indent=2))
        return 0

    try:
        health = backend.health_check()
    except OllamaError as exc:
        print(json.dumps({"backend_type": backend_type, "reachable": False, "error": str(exc)}, indent=2))
        return 1

    if "model_available" in health and not health["model_available"]:
        print(
            json.dumps(
                {
                    "backend_type": backend_type,
                    "reachable": True,
                    "model_available": False,
                    "error": f"Model '{model_name}' was not found. Pull it first with: ollama pull {model_name}",
                    "installed_models": health.get("installed_models", []),
                },
                indent=2,
            )
        )
        return 1

    print(json.dumps(health, indent=2))
    return 0


def check_local_health(config: dict) -> int:
    local_backend = create_local_backend(
        BackendConfig(
            backend_type=config["local_backend"]["backend_type"],
            model_name=config["local_backend"]["model_name"],
            quantization=config["local_backend"].get("quantization"),
            base_url=config["local_backend"].get("base_url"),
            timeout=config["local_backend"].get("timeout"),
            confidence_estimator=config["local_backend"].get("confidence_estimator", "router_v1"),
        )
    )
    return _check_backend_health(
        backend=local_backend,
        backend_type=config["local_backend"]["backend_type"],
        model_name=config["local_backend"]["model_name"],
    )


def check_edge_health(config: dict) -> int:
    edge_backend = create_edge_backend(
        BackendConfig(
            backend_type=config["edge_backend"]["backend_type"],
            model_name=config["edge_backend"]["model_name"],
            quantization=config["edge_backend"].get("quantization"),
            base_url=config["edge_backend"].get("base_url"),
            timeout=config["edge_backend"].get("timeout"),
        )
    )
    return _check_backend_health(
        backend=edge_backend,
        backend_type=config["edge_backend"]["backend_type"],
        model_name=config["edge_backend"]["model_name"],
    )


def main() -> None:
    args = build_parser().parse_args()
    config = apply_cli_overrides(load_run_config(args.config), args)

    if args.check_local_health:
        raise SystemExit(check_local_health(config))

    if args.check_edge_health:
        raise SystemExit(check_edge_health(config))

    try:
        artifacts = run_experiment(
            dataset_name=config["dataset"],
            dataset_source=config["dataset_source"],
            mode=config["mode"],
            limit=config["limit"],
            confidence_threshold=config["confidence_threshold"],
            block_sizes=config["block_sizes"],
            data_path=config["data_path"],
            log_dir=config["log_dir"],
            local_backend_type=config["local_backend"]["backend_type"],
            local_base_url=config["local_backend"]["base_url"],
            local_model_name=config["local_backend"]["model_name"],
            local_timeout=config["local_backend"]["timeout"],
            local_quantization=config["local_backend"]["quantization"],
            local_confidence_estimator=config["local_backend"]["confidence_estimator"],
            edge_backend_type=config["edge_backend"]["backend_type"],
            edge_base_url=config["edge_backend"]["base_url"],
            edge_model_name=config["edge_backend"]["model_name"],
            edge_timeout=config["edge_backend"]["timeout"],
            edge_quantization=config["edge_backend"]["quantization"],
        )
    except OllamaError as exc:
        print(f"ollama_error={exc}")
        raise SystemExit(1)

    print(f"run_id={artifacts.run_id}")
    print(f"samples={artifacts.sample_count}")
    print(f"jsonl={artifacts.jsonl_path}")
    print(f"summary_csv={artifacts.summary_csv_path}")
    print(f"metadata_json={artifacts.metadata_path}")


if __name__ == "__main__":
    main()
