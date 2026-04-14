#!/usr/bin/env bash

set -euo pipefail

DATASET="${DATASET:-mmlu}"
DATASET_SOURCE="${DATASET_SOURCE:-real}"
DATA_PATH="${DATA_PATH:-datasets/real/${DATASET}}"
MODE="${MODE:-hybrid_only}"
LIMIT="${LIMIT:-20}"

LOCAL_BACKEND_TYPE="${LOCAL_BACKEND_TYPE:-real_local_backend}"
LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://localhost:11434}"
LOCAL_MODEL_NAME="${LOCAL_MODEL_NAME:-gemma3:1b}"
LOCAL_TIMEOUT="${LOCAL_TIMEOUT:-60}"
LOCAL_QUANTIZATION="${LOCAL_QUANTIZATION:-}"
LOCAL_CONFIDENCE_ESTIMATOR="${LOCAL_CONFIDENCE_ESTIMATOR:-router_v1}"

EDGE_BACKEND_TYPE="${EDGE_BACKEND_TYPE:-ollama_edge_backend}"
EDGE_BASE_URL="${EDGE_BASE_URL:-http://localhost:11434}"
EDGE_MODEL_NAME="${EDGE_MODEL_NAME:-gemma3:4b}"
EDGE_TIMEOUT="${EDGE_TIMEOUT:-60}"
EDGE_QUANTIZATION="${EDGE_QUANTIZATION:-}"

CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.6}"
LOG_DIR="${LOG_DIR:-logs}"

cmd=(
  python -m runner.cli
  --dataset "$DATASET"
  --dataset-source "$DATASET_SOURCE"
  --data-path "$DATA_PATH"
  --mode "$MODE"
  --limit "$LIMIT"
  --confidence-threshold "$CONFIDENCE_THRESHOLD"
  --log-dir "$LOG_DIR"
  --local-backend-type "$LOCAL_BACKEND_TYPE"
  --local-base-url "$LOCAL_BASE_URL"
  --local-model-name "$LOCAL_MODEL_NAME"
  --local-timeout "$LOCAL_TIMEOUT"
  --local-confidence-estimator "$LOCAL_CONFIDENCE_ESTIMATOR"
  --edge-backend-type "$EDGE_BACKEND_TYPE"
  --edge-base-url "$EDGE_BASE_URL"
  --edge-model-name "$EDGE_MODEL_NAME"
  --edge-timeout "$EDGE_TIMEOUT"
)

if [[ -n "$LOCAL_QUANTIZATION" ]]; then
  cmd+=(--local-quantization "$LOCAL_QUANTIZATION")
fi

if [[ -n "$EDGE_QUANTIZATION" ]]; then
  cmd+=(--edge-quantization "$EDGE_QUANTIZATION")
fi

cmd+=("$@")

printf 'Running:'
for arg in "${cmd[@]}"; do
  printf ' %q' "$arg"
done
printf '\n'

"${cmd[@]}"
