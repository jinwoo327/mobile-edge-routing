#!/usr/bin/env bash

set -euo pipefail

DATASET="${DATASET:-gsm8k}"
DATASET_SOURCE="${DATASET_SOURCE:-real}"
DATA_PATH="${DATA_PATH:-datasets/real/gsm8k}"
LIMIT="${LIMIT:-100}"
LOG_DIR="${LOG_DIR:-logs}"
SKIP_FAILED_SAMPLES="${SKIP_FAILED_SAMPLES:-1}"

PROMPT_PROFILE="${PROMPT_PROFILE:-benchmark_v2}"

LOCAL_BACKEND_TYPE="${LOCAL_BACKEND_TYPE:-real_local_backend}"
LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://localhost:11434}"
LOCAL_MODEL_NAME="${LOCAL_MODEL_NAME:-gemma3:1b}"
LOCAL_TIMEOUT="${LOCAL_TIMEOUT:-120}"
LOCAL_QUANTIZATION="${LOCAL_QUANTIZATION:-}"
LOCAL_TEMPERATURE="${LOCAL_TEMPERATURE:-0}"
LOCAL_CONFIDENCE_ESTIMATOR="${LOCAL_CONFIDENCE_ESTIMATOR:-router_v1}"

EDGE_BACKEND_TYPE="${EDGE_BACKEND_TYPE:-ollama_edge_backend}"
EDGE_BASE_URL="${EDGE_BASE_URL:-http://localhost:11434}"
EDGE_MODEL_NAME="${EDGE_MODEL_NAME:-gemma3:4b}"
EDGE_TIMEOUT="${EDGE_TIMEOUT:-120}"
EDGE_QUANTIZATION="${EDGE_QUANTIZATION:-}"
EDGE_TEMPERATURE="${EDGE_TEMPERATURE:-0}"

cmd=(
  python -m runner.cli
  --dataset "$DATASET"
  --dataset-source "$DATASET_SOURCE"
  --data-path "$DATA_PATH"
  --mode local_only
  --limit "$LIMIT"
  --log-dir "$LOG_DIR"
  --prompt-profile "$PROMPT_PROFILE"
  --local-backend-type "$LOCAL_BACKEND_TYPE"
  --local-base-url "$LOCAL_BASE_URL"
  --local-model-name "$LOCAL_MODEL_NAME"
  --local-timeout "$LOCAL_TIMEOUT"
  --local-temperature "$LOCAL_TEMPERATURE"
  --local-confidence-estimator "$LOCAL_CONFIDENCE_ESTIMATOR"
  --edge-backend-type "$EDGE_BACKEND_TYPE"
  --edge-base-url "$EDGE_BASE_URL"
  --edge-model-name "$EDGE_MODEL_NAME"
  --edge-timeout "$EDGE_TIMEOUT"
  --edge-temperature "$EDGE_TEMPERATURE"
)

if [[ "$SKIP_FAILED_SAMPLES" == "1" ]]; then
  cmd+=(--skip-failed-samples)
fi

if [[ -n "$LOCAL_QUANTIZATION" ]]; then
  cmd+=(--local-quantization "$LOCAL_QUANTIZATION")
fi

if [[ -n "$EDGE_QUANTIZATION" ]]; then
  cmd+=(--edge-quantization "$EDGE_QUANTIZATION")
fi

printf 'Running:'
for arg in "${cmd[@]}"; do
  printf ' %q' "$arg"
done
printf '\n'

set +e
output="$("${cmd[@]}" 2>&1)"
status=$?
set -e

printf '%s\n' "$output"

if [[ $status -ne 0 ]]; then
  echo
  echo "Run failed. Check the newest error file with:"
  echo "ls -lt ${LOG_DIR}/run-*-error.json | head"
  exit "$status"
fi
