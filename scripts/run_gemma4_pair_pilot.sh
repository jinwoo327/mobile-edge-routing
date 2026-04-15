#!/usr/bin/env bash

set -euo pipefail

DATASET="${DATASET:-mmlu}"
DATASET_SOURCE="${DATASET_SOURCE:-real}"
DATA_PATH="${DATA_PATH:-datasets/real/${DATASET}}"
MODE="${MODE:-hybrid_only}"
LIMIT="${LIMIT:-10}"
PROMPT_PROFILE="${PROMPT_PROFILE:-benchmark_v2}"
LOG_DIR="${LOG_DIR:-logs}"

LOCAL_MODEL_NAME="${LOCAL_MODEL_NAME:-gemma4:e2b}"
EDGE_MODEL_NAME="${EDGE_MODEL_NAME:-gemma4:e4b}"
LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://localhost:11434}"
EDGE_BASE_URL="${EDGE_BASE_URL:-http://localhost:11434}"

cmd=(
  python -m runner.cli
  --dataset "$DATASET"
  --dataset-source "$DATASET_SOURCE"
  --data-path "$DATA_PATH"
  --mode "$MODE"
  --limit "$LIMIT"
  --prompt-profile "$PROMPT_PROFILE"
  --log-dir "$LOG_DIR"
  --local-backend-type real_local_backend
  --local-model-name "$LOCAL_MODEL_NAME"
  --local-base-url "$LOCAL_BASE_URL"
  --local-timeout 120
  --local-temperature 0
  --local-confidence-estimator router_v1
  --edge-backend-type ollama_edge_backend
  --edge-model-name "$EDGE_MODEL_NAME"
  --edge-base-url "$EDGE_BASE_URL"
  --edge-timeout 120
  --edge-temperature 0
)

if [[ "$DATASET" == "gsm8k" ]]; then
  cmd+=(--skip-failed-samples)
fi

if [[ "$MODE" == "hybrid_only" ]]; then
  cmd+=(--confidence-threshold 0.6)
fi

printf 'Running:'
for arg in "${cmd[@]}"; do
  printf ' %q' "$arg"
done
printf '\n'

"${cmd[@]}"
