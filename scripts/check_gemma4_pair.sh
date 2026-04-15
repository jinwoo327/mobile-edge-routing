#!/usr/bin/env bash

set -euo pipefail

LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://localhost:11434}"
EDGE_BASE_URL="${EDGE_BASE_URL:-http://localhost:11434}"
LOCAL_MODEL_NAME="${LOCAL_MODEL_NAME:-gemma4:e2b}"
EDGE_MODEL_NAME="${EDGE_MODEL_NAME:-gemma4:e4b}"

python -m runner.cli \
  --local-backend-type real_local_backend \
  --local-model-name "$LOCAL_MODEL_NAME" \
  --local-base-url "$LOCAL_BASE_URL" \
  --check-local-health

python -m runner.cli \
  --edge-backend-type ollama_edge_backend \
  --edge-model-name "$EDGE_MODEL_NAME" \
  --edge-base-url "$EDGE_BASE_URL" \
  --check-edge-health
