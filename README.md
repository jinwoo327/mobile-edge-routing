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
python -m runner.cli --dataset all --dataset-source fixture --mode hybrid_only --limit 8
```

You can also use the helper script:

```bash
bash scripts/smoke_test.sh
```

## CLI Example

```bash
python -m runner.cli \
  --dataset mmlu \
  --dataset-source fixture \
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
- `--local-confidence-estimator router_v1`
- `--edge-backend-type mock_edge_backend`
- `--edge-backend-type ollama_edge_backend`
- `--edge-base-url http://localhost:11434`
- `--edge-model-name gemma3:4b`
- `--edge-timeout 60`
- `--edge-quantization Q4_K_M`
- `--local-model-name gemma3:1b`
- `--local-quantization Q4_K_M`

Quantization is tracked as metadata only in this version. It is not hardcoded into routing logic, so later experiments can align local and edge quantization for fair comparisons.

For hybrid routing, the local confidence estimator is also configurable:

- `format_heuristic`: legacy output-shape heuristic
- `router_v1`: improved single-pass heuristic that combines answer format with question complexity penalties

Important dataset options:

- `--dataset gsm8k`
- `--dataset mmlu`
- `--dataset-source fixture`
- `--dataset-source real`
- `--data-path datasets/real/gsm8k`
- `--data-path datasets/real/mmlu`

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

## Real Dataset Layout

Fixture mode and real mode are separate:

- fixture mode loads the tiny bundled smoke-test files
- real mode loads larger local files or directories that you provide

Default real dataset locations:

- `datasets/real/gsm8k/`
- `datasets/real/mmlu/`

You can also override these with `--data-path`.

Automatic download helper:

```bash
pip install datasets
python scripts/download_real_datasets.py --dataset gsm8k
python scripts/download_real_datasets.py --dataset mmlu
```

This helper removes the repo root from `sys.path` before importing Hugging Face `datasets`, so it avoids the local `datasets/` package name collision.

### Real GSM8K

Expected input:

- a `.jsonl` file
- a `.json` file containing a list of records
- or a directory containing one or more `.jsonl` / `.json` files

Expected record fields:

- `question`
- `answer`
- optional `id`

Recommended placement:

- `datasets/real/gsm8k/test.jsonl`

### Real MMLU

Expected input:

- a `.csv` file
- a `.jsonl` file
- a `.json` file containing a list of records
- or a directory containing one or more `.csv` / `.jsonl` / `.json` files

Expected record fields:

- `question`
- either `choices` or the four columns `A`, `B`, `C`, `D`
- `answer` or `target`
- optional `id`
- optional `subject`

Recommended placement:

- `datasets/real/mmlu/<subject>.csv`
- or `python scripts/download_real_datasets.py --dataset mmlu`, which writes `datasets/real/mmlu/all_test.jsonl` by default

CSV format example:

```text
question,A,B,C,D,answer
2+2=?,3,4,5,6,B
```

When loading MMLU from a directory of CSV files, the filename stem is used as `subject` metadata if the file does not already provide it.

## Dataset Examples

4-sample fixture GSM8K smoke test:

```bash
python -m runner.cli \
  --dataset gsm8k \
  --dataset-source fixture \
  --mode local_only \
  --limit 4
```

20-sample real GSM8K experiment:

```bash
python -m runner.cli \
  --dataset gsm8k \
  --dataset-source real \
  --data-path datasets/real/gsm8k \
  --mode local_only \
  --limit 20
```

Shortcut script:

```bash
bash scripts/run_real_experiment.sh
```

Examples:

```bash
DATASET=mmlu bash scripts/run_real_experiment.sh
MODE=hybrid_only LOCAL_BACKEND_TYPE=real_local_backend EDGE_BACKEND_TYPE=ollama_edge_backend bash scripts/run_real_experiment.sh
LIMIT=50 DATASET=gsm8k bash scripts/run_real_experiment.sh
```

20-sample real MMLU experiment:

```bash
python -m runner.cli \
  --dataset mmlu \
  --dataset-source real \
  --data-path datasets/real/mmlu \
  --mode local_only \
  --limit 20
```

Run all three modes at once with configurable dataset and sample count:

```bash
bash scripts/run_all_modes.sh
```

Examples:

```bash
DATASET=mmlu LIMIT=20 bash scripts/run_all_modes.sh
DATASET=gsm8k LIMIT=50 bash scripts/run_all_modes.sh
DATASET_SOURCE=fixture DATASET=gsm8k LIMIT=4 bash scripts/run_all_modes.sh
CONFIDENCE_THRESHOLD=0.7 DATASET=mmlu LIMIT=20 bash scripts/run_all_modes.sh
```

`run_all_modes.sh` now also writes a combined comparison CSV to:

- `logs/compare/<dataset>-<dataset_source>-<limit>-compare.csv`

and, if `matplotlib` is installed, a quality-latency plot to:

- `logs/compare/<dataset>-<dataset_source>-<limit>-compare.png`

You can control warm-up exclusion for latency comparison with:

```bash
EXCLUDE_FIRST_N_LATENCY=1 bash scripts/run_all_modes.sh
```

Compare completed runs into one CSV:

```bash
python scripts/compare_runs.py \
  --run-ids 20260414-191938 20260414-192103 20260414-192332 \
  --output-csv logs/mmlu-20-compare.csv
```

Quality-latency scatter plot:

```bash
python scripts/compare_runs.py \
  --run-ids 20260414-191938 20260414-192103 20260414-192332 \
  --output-csv logs/mmlu-20-compare.csv \
  --plot-path logs/mmlu-20-compare.png
```

Equivalent config examples:

- `configs/fixture_gsm8k.json`
- `configs/real_gsm8k.json`
- `configs/fixture_mmlu.json`
- `configs/real_mmlu.json`

If the requested `--limit` is larger than the available dataset size, the runner prints a warning and evaluates all available samples.

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

## Router Analysis

Threshold sweep for hybrid routing:

```bash
python scripts/threshold_sweep.py \
  --dataset mmlu \
  --dataset-source real \
  --data-path datasets/real/mmlu \
  --limit 50 \
  --local-backend-type real_local_backend \
  --edge-backend-type ollama_edge_backend \
  --local-confidence-estimator router_v1
```

Pareto CSV and plot generation from a sweep:

```bash
python scripts/pareto_analyze.py \
  --input-csv logs/sweeps/sweep-<id>.csv \
  --plot-path logs/sweeps/sweep-<id>-pareto.png
```

The Pareto analysis treats:

- `accuracy` as quality
- `mean_latency_ms` as latency
- `edge_rate` as a communication/cost proxy

This keeps the current repo explicit about the fact that real battery measurement is still not implemented.

## Not Yet Implemented

- Android or smartphone-local inference
- real battery instrumentation on device
- real mobile network measurement
- multi-turn interactions
- deployment packaging for phones
