# Mobile Edge Routing

Minimal research scaffold for single-turn edge-device collaborative LLM inference experiments.

This first version compares three fixed execution modes:

- `local_only`
- `edge_only`
- `hybrid_only`

Current scope:

- single-turn inference only
- datasets: MMLU and GSM8K
- selectable local backend: mock or Ollama-backed Gemma 3 1B
- selectable edge backend: mock or Ollama-backed Gemma 3 4B
- no smartphone integration yet
- no real battery or network measurement yet

The repository is intentionally lightweight so a student team can extend it without untangling framework-heavy code.

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
├── datasets/
├── evaluation/
├── logs/
├── models/
├── monitoring/
├── routing/
├── runner/
├── scripts/
└── tests/
```

## Quick Start

1. Create a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Run the unit tests.

```bash
python -m unittest discover -s tests
```

4. Run a smoke test with bundled mock data.

```bash
python -m runner.cli --dataset all --mode hybrid_only --limit 8
```

You can also use the helper script:

```bash
bash scripts/smoke_test.sh
```

## CLI Example

```bash
python -m runner.cli \
  --dataset mmlu \
  --mode hybrid_only \
  --limit 4
```

Outputs:

- per-sample JSONL log in `logs/`
- grouped CSV summary for block sizes `20`, `50`, and `100`
- run metadata JSON in `logs/`

## Configuration

The runner accepts either direct CLI flags or a JSON config file.

Example:

```bash
python -m runner.cli --config configs/real_local_real_edge.json
```

Important backend options:

- `--local-backend-type mock_local_backend`
- `--local-backend-type real_local_backend`
- `--local-base-url http://localhost:11434`
- `--local-model-name gemma3:1b`
- `--local-timeout 60`
- `--local-quantization Q4_K_M`
- `--edge-backend-type mock_edge_backend`
- `--edge-backend-type ollama_edge_backend`
- `--edge-base-url http://localhost:11434`
- `--edge-model-name gemma3:4b`
- `--edge-timeout 60`
- `--edge-quantization Q4_K_M`
- `--local-model-name gemma3:1b`
- `--local-quantization Q4_K_M`

Quantization is tracked as metadata only in this version. It is not hardcoded into routing logic, so later experiments can align local and edge quantization for fair comparisons.

## Ollama Setup

1. Install Ollama.

Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

macOS:

- install the Ollama app from the official download page and launch it once

Windows:

- install the Ollama application from the official download page; it serves the API on `http://localhost:11434`

2. Pull the edge model.

```bash
ollama pull gemma3:1b
ollama pull gemma3:4b
```

3. Start the Ollama service if it is not already running.

```bash
ollama serve
```

4. Verify connectivity and model availability.

Local:

```bash
python -m runner.cli \
  --local-backend-type real_local_backend \
  --local-model-name gemma3:1b \
  --check-local-health
```

Edge:

```bash
python -m runner.cli \
  --edge-backend-type ollama_edge_backend \
  --edge-model-name gemma3:4b \
  --check-edge-health
```

You can also use:

```bash
bash scripts/check_local_ollama.sh
bash scripts/check_ollama.sh
```

5. Run `local_only` with the real local backend.

```bash
python -m runner.cli \
  --dataset all \
  --mode local_only \
  --limit 4 \
  --local-backend-type real_local_backend \
  --local-model-name gemma3:1b \
  --local-base-url http://localhost:11434
```

6. Run a small pilot experiment with mock local plus real Ollama edge.

```bash
python -m runner.cli \
  --dataset all \
  --mode hybrid_only \
  --limit 4 \
  --edge-backend-type ollama_edge_backend \
  --edge-model-name gemma3:4b \
  --edge-base-url http://localhost:11434
```

7. Run `hybrid_only` with real local plus real edge.

```bash
python -m runner.cli \
  --dataset all \
  --mode hybrid_only \
  --limit 4 \
  --local-backend-type real_local_backend \
  --local-model-name gemma3:1b \
  --edge-backend-type ollama_edge_backend \
  --edge-model-name gemma3:4b
```

8. Run edge-only with the real Ollama edge backend.

```bash
python -m runner.cli \
  --dataset gsm8k \
  --mode edge_only \
  --limit 4 \
  --edge-backend-type ollama_edge_backend \
  --edge-model-name gemma3:4b
```

Smoke-test helpers:

```bash
bash scripts/smoke_local_real.sh
bash scripts/smoke_hybrid_real.sh
```

## Bundled Mock Data

The repo ships with tiny JSONL fixture files so the pipeline runs immediately:

- `datasets/fixtures/mmlu_mock.jsonl`
- `datasets/fixtures/gsm8k_mock.jsonl`

These are not benchmark distributions. They are just smoke-test inputs.

Some fixture samples include mock metadata such as `mock_local_prediction` and `mock_local_confidence`. That keeps routing behavior deterministic during early development. Real experiments should replace the mock backends and point the loaders at real dataset files.

## Fairness Tracking

Run metadata now records:

- local backend type and model name
- local base URL and timeout
- edge backend type, base URL, and model name
- local quantization
- edge quantization

This keeps fairness-related settings explicit without pushing battery or quantization awareness into the hybrid routing logic itself.

Per-sample JSONL and grouped CSV outputs also carry local and edge quantization fields so later fair-comparison runs can be filtered or compared consistently.

## Confidence Notes

Hybrid routing still uses the same simple local-first rule:

- run the local backend first
- use the backend confidence score
- fall back to edge when confidence is below the threshold

When Ollama exposes usable token log probabilities, the backend converts them into a simple normalized confidence score. When that is unavailable, the repository uses a documented fallback confidence estimator based on answer-format compliance:

- multiple choice: higher confidence for a clean single-letter answer
- GSM8K-style math: higher confidence for a clean standalone numeric answer

The confidence source is logged per sample.

## Not Yet Implemented

- Android or smartphone-local inference
- real battery instrumentation on device
- real mobile network measurement
- multi-turn interactions
- deployment packaging for phones
