#!/usr/bin/env bash

python -m runner.cli \
  --edge-backend-type ollama_edge_backend \
  --edge-model-name gemma3:4b \
  --edge-base-url http://localhost:11434 \
  --check-edge-health
