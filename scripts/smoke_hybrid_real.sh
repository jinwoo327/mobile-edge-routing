#!/usr/bin/env bash

python -m runner.cli \
  --dataset all \
  --mode hybrid_only \
  --limit 4 \
  --local-backend-type real_local_backend \
  --local-model-name gemma3:1b \
  --local-base-url http://localhost:11434 \
  --edge-backend-type ollama_edge_backend \
  --edge-model-name gemma3:4b \
  --edge-base-url http://localhost:11434
