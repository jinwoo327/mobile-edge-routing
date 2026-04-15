#!/usr/bin/env bash

set -euo pipefail

DATASET="${DATASET:-mmlu}"
DATASET_SOURCE="${DATASET_SOURCE:-real}"
DATA_PATH="${DATA_PATH:-datasets/real/${DATASET}}"
LIMIT="${LIMIT:-20}"
LOG_DIR="${LOG_DIR:-logs}"
COMPARE_DIR="${COMPARE_DIR:-${LOG_DIR}/compare}"
EXCLUDE_FIRST_N_LATENCY="${EXCLUDE_FIRST_N_LATENCY:-1}"
SKIP_FAILED_SAMPLES="${SKIP_FAILED_SAMPLES:-0}"
PROMPT_PROFILE="${PROMPT_PROFILE:-benchmark_v2}"

LOCAL_BACKEND_TYPE="${LOCAL_BACKEND_TYPE:-real_local_backend}"
LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://localhost:11434}"
LOCAL_MODEL_NAME="${LOCAL_MODEL_NAME:-gemma3:1b}"
LOCAL_TIMEOUT="${LOCAL_TIMEOUT:-60}"
LOCAL_QUANTIZATION="${LOCAL_QUANTIZATION:-}"
LOCAL_TEMPERATURE="${LOCAL_TEMPERATURE:-0}"
LOCAL_CONFIDENCE_ESTIMATOR="${LOCAL_CONFIDENCE_ESTIMATOR:-router_v1}"

EDGE_BACKEND_TYPE="${EDGE_BACKEND_TYPE:-ollama_edge_backend}"
EDGE_BASE_URL="${EDGE_BASE_URL:-http://localhost:11434}"
EDGE_MODEL_NAME="${EDGE_MODEL_NAME:-gemma3:4b}"
EDGE_TIMEOUT="${EDGE_TIMEOUT:-60}"
EDGE_QUANTIZATION="${EDGE_QUANTIZATION:-}"
EDGE_TEMPERATURE="${EDGE_TEMPERATURE:-0}"

CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.6}"

RUN_IDS=()

run_mode() {
  local mode="$1"
  local output
  local -a cmd=(
    python -m runner.cli
    --dataset "$DATASET"
    --dataset-source "$DATASET_SOURCE"
    --data-path "$DATA_PATH"
    --mode "$mode"
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

  if [[ "$mode" == "hybrid_only" ]]; then
    cmd+=(--confidence-threshold "$CONFIDENCE_THRESHOLD")
  fi

  if [[ -n "$LOCAL_QUANTIZATION" ]]; then
    cmd+=(--local-quantization "$LOCAL_QUANTIZATION")
  fi

  if [[ -n "$EDGE_QUANTIZATION" ]]; then
    cmd+=(--edge-quantization "$EDGE_QUANTIZATION")
  fi

  printf '\n[%s] Running:' "$mode"
  for arg in "${cmd[@]}"; do
    printf ' %q' "$arg"
  done
  printf '\n'

  output="$("${cmd[@]}")"
  printf '%s\n' "$output"

  local run_id
  run_id="$(printf '%s\n' "$output" | awk -F= '/^run_id=/{print $2}' | tail -n 1)"
  if [[ -z "$run_id" ]]; then
    echo "Failed to extract run_id for mode $mode" >&2
    exit 1
  fi
  RUN_IDS+=("$run_id")
}

run_mode local_only
run_mode edge_only
run_mode hybrid_only

mkdir -p "$COMPARE_DIR"
COMPARE_CSV="${COMPARE_DIR}/${DATASET}-${DATASET_SOURCE}-${LIMIT}-compare.csv"
COMPARE_PLOT="${COMPARE_DIR}/${DATASET}-${DATASET_SOURCE}-${LIMIT}-compare.png"

printf '\n[compare] Building comparison CSV in %s\n' "$COMPARE_CSV"
python scripts/compare_runs.py \
  --run-ids "${RUN_IDS[@]}" \
  --logs-dir "$LOG_DIR" \
  --output-csv "$COMPARE_CSV" \
  --exclude-first-n-latency "$EXCLUDE_FIRST_N_LATENCY"

if python -c "import matplotlib" >/dev/null 2>&1; then
  printf '[compare] Building comparison plot in %s\n' "$COMPARE_PLOT"
  python scripts/compare_runs.py \
    --run-ids "${RUN_IDS[@]}" \
    --logs-dir "$LOG_DIR" \
    --output-csv "$COMPARE_CSV" \
    --plot-path "$COMPARE_PLOT" \
    --exclude-first-n-latency "$EXCLUDE_FIRST_N_LATENCY" \
    --plot-latency-stat median
else
  printf '[compare] Skipping plot because matplotlib is not installed.\n'
fi
