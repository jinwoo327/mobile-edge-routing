#!/usr/bin/env bash

python -m runner.cli \
  --dataset all \
  --mode local_only \
  --limit 4 \
  --local-backend-type real_local_backend \
  --local-model-name gemma3:1b \
  --local-base-url http://localhost:11434
