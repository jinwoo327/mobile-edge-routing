from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "datasets" / "real"


def _remove_repo_from_sys_path() -> None:
    repo_entries = {str(REPO_ROOT), ""}
    sys.path[:] = [entry for entry in sys.path if entry not in repo_entries]


def _load_hf_datasets():
    _remove_repo_from_sys_path()
    try:
        module = importlib.import_module("datasets")
    except ImportError as exc:
        raise SystemExit(
            "Hugging Face 'datasets' is not installed. Install it with: pip install datasets"
        ) from exc
    return module


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")
            count += 1
    return count


def download_gsm8k(split: str, output_root: Path) -> Path:
    datasets_module = _load_hf_datasets()
    dataset = datasets_module.load_dataset("openai/gsm8k", "main", split=split)
    output_path = output_root / "gsm8k" / f"{split}.jsonl"

    def _rows() -> Iterable[dict]:
        for index, row in enumerate(dataset):
            yield {
                "id": f"gsm8k-{split}-{index}",
                "question": row["question"],
                "answer": row["answer"],
            }

    count = _write_jsonl(output_path, _rows())
    print(f"wrote {count} rows to {output_path}")
    return output_path


def download_mmlu(split: str, subset: str, output_root: Path) -> Path:
    datasets_module = _load_hf_datasets()
    dataset = datasets_module.load_dataset("cais/mmlu", subset, split=split)
    output_path = output_root / "mmlu" / f"{subset}_{split}.jsonl"

    def _rows() -> Iterable[dict]:
        for index, row in enumerate(dataset):
            answer = row["answer"]
            if isinstance(answer, int):
                answer = chr(ord("A") + answer)

            payload = {
                "id": f"mmlu-{subset}-{split}-{index}",
                "question": row["question"],
                "choices": row["choices"],
                "answer": answer,
            }
            if row.get("subject"):
                payload["subject"] = row["subject"]
            yield payload

    count = _write_jsonl(output_path, _rows())
    print(f"wrote {count} rows to {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download real GSM8K or MMLU data into datasets/real/.")
    parser.add_argument("--dataset", choices=["gsm8k", "mmlu", "all"], required=True)
    parser.add_argument("--split", default="test", help="Dataset split to download. Default: test")
    parser.add_argument(
        "--mmlu-subset",
        default="all",
        help="MMLU subset/config name. Use 'all' for a combined set or a subject config if needed.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory to write normalized real datasets into.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)

    if args.dataset in {"gsm8k", "all"}:
        download_gsm8k(split=args.split, output_root=output_root)

    if args.dataset in {"mmlu", "all"}:
        download_mmlu(split=args.split, subset=args.mmlu_subset, output_root=output_root)


if __name__ == "__main__":
    main()
