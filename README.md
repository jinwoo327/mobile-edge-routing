# Mobile Edge Routing

Minimal research scaffold for single-turn edge-device collaborative LLM inference experiments.

This first version compares three fixed execution modes:

- `local_only`
- `edge_only`
- `hybrid_only`

Current scope:

- single-turn inference only
- datasets: MMLU and GSM8K
- mock local backend standing in for Gemma 3 1B
- mock edge backend standing in for Gemma 3 4B
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
  --limit 4 \
  --confidence-threshold 0.60
```

Outputs:

- per-sample JSONL log in `logs/`
- grouped CSV summary for block sizes `20`, `50`, and `100`

## Bundled Mock Data

The repo ships with tiny JSONL fixture files so the pipeline runs immediately:

- `datasets/fixtures/mmlu_mock.jsonl`
- `datasets/fixtures/gsm8k_mock.jsonl`

These are not benchmark distributions. They are just smoke-test inputs.

Some fixture samples include mock metadata such as `mock_local_prediction` and `mock_local_confidence`. That keeps routing behavior deterministic during early development. Real experiments should replace the mock backends and point the loaders at real dataset files.

## Extending This Foundation

- Replace `models.backends.MockLocalBackend` with a real on-device Gemma 3 1B backend.
- Replace `models.backends.MockEdgeBackend` with an Ollama-backed Gemma 3 4B client.
- Keep battery and network measurement in `monitoring/`, outside routing logic.
- Keep `routing/` focused on execution decisions only.

## Suggested Next Steps

- add real dataset download/preprocessing scripts
- add Ollama edge client
- add calibrated confidence estimation
- add richer latency and energy accounting
