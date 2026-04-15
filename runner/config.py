from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_RUN_CONFIG: dict[str, Any] = {
    "dataset": "all",
    "dataset_source": "fixture",
    "data_path": None,
    "mode": "hybrid_only",
    "limit": None,
    "confidence_threshold": 0.6,
    "skip_failed_samples": False,
    "prompt_profile": "benchmark_v2",
    "block_sizes": [20, 50, 100],
    "log_dir": "logs",
    "local_backend": {
        "backend_type": "mock_local_backend",
        "base_url": "http://localhost:11434",
        "model_name": "gemma3:1b",
        "timeout": 60.0,
        "quantization": None,
        "confidence_estimator": "router_v1",
        "temperature": 0.0,
    },
    "edge_backend": {
        "backend_type": "mock_edge_backend",
        "base_url": "http://localhost:11434",
        "model_name": "gemma3:4b",
        "timeout": 60.0,
        "quantization": None,
        "temperature": 0.0,
    },
}


def load_run_config(path: str | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_RUN_CONFIG)
    if not path:
        return config

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    return _deep_merge(config, loaded)


def apply_cli_overrides(config: dict[str, Any], args: Any) -> dict[str, Any]:
    resolved = deepcopy(config)

    scalar_mappings = {
        "dataset": "dataset",
        "dataset_source": "dataset_source",
        "data_path": "data_path",
        "mode": "mode",
        "limit": "limit",
        "confidence_threshold": "confidence_threshold",
        "skip_failed_samples": "skip_failed_samples",
        "prompt_profile": "prompt_profile",
        "block_sizes": "block_sizes",
        "log_dir": "log_dir",
    }
    for arg_name, config_key in scalar_mappings.items():
        value = getattr(args, arg_name)
        if value is not None:
            resolved[config_key] = value

    local_overrides = {
        "local_backend_type": "backend_type",
        "local_base_url": "base_url",
        "local_model_name": "model_name",
        "local_timeout": "timeout",
        "local_quantization": "quantization",
        "local_confidence_estimator": "confidence_estimator",
        "local_temperature": "temperature",
    }
    for arg_name, config_key in local_overrides.items():
        value = getattr(args, arg_name)
        if value is not None:
            resolved["local_backend"][config_key] = value

    edge_overrides = {
        "edge_backend_type": "backend_type",
        "edge_base_url": "base_url",
        "edge_model_name": "model_name",
        "edge_timeout": "timeout",
        "edge_quantization": "quantization",
        "edge_temperature": "temperature",
    }
    for arg_name, config_key in edge_overrides.items():
        value = getattr(args, arg_name)
        if value is not None:
            resolved["edge_backend"][config_key] = value

    if resolved.get("prompt_profile") is not None:
        resolved["local_backend"]["prompt_profile"] = resolved["prompt_profile"]
        resolved["edge_backend"]["prompt_profile"] = resolved["prompt_profile"]

    return resolved


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
