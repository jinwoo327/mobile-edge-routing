from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from datasets.common import DatasetSample


DatasetSource = str


@dataclass(slots=True)
class DatasetLoadResult:
    dataset_name: str
    dataset_source: DatasetSource
    samples: list[DatasetSample]
    available_count: int
    requested_limit: int | None
    resolved_paths: list[str] = field(default_factory=list)
    warning: str | None = None


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_json(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        for item in payload:
            yield item
        return
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
        for item in payload["data"]:
            yield item
        return
    raise ValueError(f"Unsupported JSON dataset payload in {path}. Expected a list of records.")


def _read_csv(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield dict(row)


def _read_records(path: Path) -> Iterable[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if suffix == ".json":
        return _read_json(path)
    if suffix == ".csv":
        return _read_csv(path)
    raise ValueError(f"Unsupported dataset file format: {path}")


def _default_fixture_path(dataset_name: str) -> Path:
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    return fixture_dir / f"{dataset_name}_mock.jsonl"


def _default_real_path(dataset_name: str) -> Path:
    real_dir = Path(__file__).resolve().parent / "real"
    return real_dir / dataset_name


def _resolve_dataset_paths(dataset_name: str, dataset_source: DatasetSource, path: str | Path | None) -> list[Path]:
    if dataset_source not in {"fixture", "real"}:
        raise ValueError(f"Unsupported dataset source: {dataset_source}")

    if dataset_source == "fixture":
        if path:
            return [Path(path)]
        return [_default_fixture_path(dataset_name)]

    target = Path(path) if path else _default_real_path(dataset_name)
    if not target.exists():
        raise FileNotFoundError(
            f"Real dataset path not found for {dataset_name}: {target}. "
            f"Pass --data-path explicitly or place the dataset under {_default_real_path(dataset_name)}"
        )

    if target.is_file():
        return [target]

    supported_paths = sorted(
        candidate
        for candidate in target.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".jsonl", ".json", ".csv"}
    )
    if not supported_paths:
        raise FileNotFoundError(
            f"No supported dataset files were found under {target}. "
            "Expected .jsonl, .json, or .csv files."
        )
    return supported_paths


class MMLULoader:
    name = "mmlu"

    def load(
        self,
        path: str | Path | None = None,
        limit: int | None = None,
        dataset_source: DatasetSource = "fixture",
    ) -> DatasetLoadResult:
        source_paths = _resolve_dataset_paths(self.name, dataset_source=dataset_source, path=path)
        samples: list[DatasetSample] = []

        for source_path in source_paths:
            for index, record in enumerate(_read_records(source_path)):
                choices = self._extract_choices(record)
                answer_letter, answer_text = self._extract_answer(record, choices)
                metadata = {
                    key: value
                    for key, value in record.items()
                    if key
                    not in {
                        "id",
                        "question",
                        "choices",
                        "answer",
                        "target",
                        "A",
                        "B",
                        "C",
                        "D",
                    }
                }
                if dataset_source == "real" and "subject" not in metadata:
                    metadata["subject"] = source_path.stem
                samples.append(
                    DatasetSample(
                        sample_id=str(record.get("id", f"{source_path.stem}-{index}")),
                        dataset=self.name,
                        task_type="multiple_choice",
                        question=str(record["question"]),
                        choices=choices,
                        answer=answer_letter,
                        answer_text=answer_text,
                        metadata=metadata,
                    )
                )

        return _build_load_result(
            dataset_name=self.name,
            dataset_source=dataset_source,
            samples=samples,
            limit=limit,
            resolved_paths=source_paths,
        )

    @staticmethod
    def _extract_choices(record: dict) -> list[str]:
        if "choices" in record:
            return [str(choice) for choice in record["choices"]]
        if all(key in record for key in ("A", "B", "C", "D")):
            return [str(record["A"]), str(record["B"]), str(record["C"]), str(record["D"])]
        raise ValueError("MMLU record must contain either 'choices' or A/B/C/D fields.")

    @staticmethod
    def _extract_answer(record: dict, choices: list[str]) -> tuple[str, str | None]:
        raw_answer = record.get("answer", record.get("target"))
        if raw_answer is None:
            raise ValueError("MMLU record is missing 'answer' or 'target'.")

        if isinstance(raw_answer, int):
            answer_index = raw_answer
        elif isinstance(raw_answer, str) and raw_answer.strip().isdigit():
            answer_index = int(raw_answer.strip())
        else:
            normalized = str(raw_answer).strip().upper()
            if normalized in {"A", "B", "C", "D"}:
                answer_index = ord(normalized) - ord("A")
            else:
                answer_text = str(raw_answer).strip()
                if answer_text in choices:
                    answer_index = choices.index(answer_text)
                else:
                    raise ValueError(f"Unsupported MMLU answer format: {raw_answer!r}")

        if answer_index < 0 or answer_index >= len(choices):
            raise ValueError(f"MMLU answer index out of range: {answer_index}")

        answer_letter = chr(ord("A") + answer_index)
        return answer_letter, choices[answer_index]


class GSM8KLoader:
    name = "gsm8k"

    def load(
        self,
        path: str | Path | None = None,
        limit: int | None = None,
        dataset_source: DatasetSource = "fixture",
    ) -> DatasetLoadResult:
        source_paths = _resolve_dataset_paths(self.name, dataset_source=dataset_source, path=path)
        samples: list[DatasetSample] = []

        for source_path in source_paths:
            for index, record in enumerate(_read_records(source_path)):
                raw_answer = str(record["answer"])
                metadata = {
                    key: value for key, value in record.items() if key not in {"id", "question", "answer"}
                }
                samples.append(
                    DatasetSample(
                        sample_id=str(record.get("id", f"{source_path.stem}-{index}")),
                        dataset=self.name,
                        task_type="free_form",
                        question=str(record["question"]),
                        answer=self._extract_final_answer(raw_answer),
                        answer_text=raw_answer,
                        metadata=metadata,
                    )
                )

        return _build_load_result(
            dataset_name=self.name,
            dataset_source=dataset_source,
            samples=samples,
            limit=limit,
            resolved_paths=source_paths,
        )

    @staticmethod
    def _extract_final_answer(raw_answer: str) -> str:
        marker_match = re.search(r"####\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)", raw_answer)
        if marker_match:
            return marker_match.group(1).replace(",", "")

        number_matches = re.findall(r"[-+]?[0-9][0-9,]*(?:\.[0-9]+)?", raw_answer)
        if number_matches:
            return number_matches[-1].replace(",", "")

        return raw_answer.strip()


def _build_load_result(
    dataset_name: str,
    dataset_source: DatasetSource,
    samples: list[DatasetSample],
    limit: int | None,
    resolved_paths: list[Path],
) -> DatasetLoadResult:
    available_count = len(samples)
    selected_samples = samples[:limit] if limit is not None else samples
    warning = None
    if limit is not None and limit > available_count:
        warning = (
            f"Requested limit {limit} for {dataset_name} ({dataset_source}) but only "
            f"{available_count} samples were available."
        )
    return DatasetLoadResult(
        dataset_name=dataset_name,
        dataset_source=dataset_source,
        samples=selected_samples,
        available_count=available_count,
        requested_limit=limit,
        resolved_paths=[str(path) for path in resolved_paths],
        warning=warning,
    )


def load_dataset(
    dataset_name: str,
    path: str | Path | None = None,
    limit: int | None = None,
    dataset_source: DatasetSource = "fixture",
) -> DatasetLoadResult:
    normalized = dataset_name.lower()
    if normalized == "mmlu":
        return MMLULoader().load(path=path, limit=limit, dataset_source=dataset_source)
    if normalized == "gsm8k":
        return GSM8KLoader().load(path=path, limit=limit, dataset_source=dataset_source)
    raise ValueError(f"Unsupported dataset: {dataset_name}")
