from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from datasets.common import DatasetSample


def _read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _default_fixture_path(dataset_name: str) -> Path:
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    return fixture_dir / f"{dataset_name}_mock.jsonl"


class MMLULoader:
    name = "mmlu"

    def load(self, path: str | Path | None = None, limit: int | None = None) -> list[DatasetSample]:
        source_path = Path(path) if path else _default_fixture_path(self.name)
        samples: list[DatasetSample] = []

        for index, record in enumerate(_read_jsonl(source_path)):
            choices = self._extract_choices(record)
            answer_letter, answer_text = self._extract_answer(record, choices)
            metadata = {
                key: value
                for key, value in record.items()
                if key not in {"id", "question", "choices", "answer", "target", "A", "B", "C", "D"}
            }
            samples.append(
                DatasetSample(
                    sample_id=str(record.get("id", f"mmlu-{index}")),
                    dataset=self.name,
                    task_type="multiple_choice",
                    question=record["question"],
                    choices=choices,
                    answer=answer_letter,
                    answer_text=answer_text,
                    metadata=metadata,
                )
            )
            if limit is not None and len(samples) >= limit:
                break

        return samples

    @staticmethod
    def _extract_choices(record: dict) -> list[str]:
        if "choices" in record:
            return list(record["choices"])
        if all(key in record for key in ("A", "B", "C", "D")):
            return [record["A"], record["B"], record["C"], record["D"]]
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

    def load(self, path: str | Path | None = None, limit: int | None = None) -> list[DatasetSample]:
        source_path = Path(path) if path else _default_fixture_path(self.name)
        samples: list[DatasetSample] = []

        for index, record in enumerate(_read_jsonl(source_path)):
            raw_answer = str(record["answer"])
            metadata = {
                key: value for key, value in record.items() if key not in {"id", "question", "answer"}
            }
            samples.append(
                DatasetSample(
                    sample_id=str(record.get("id", f"gsm8k-{index}")),
                    dataset=self.name,
                    task_type="free_form",
                    question=record["question"],
                    answer=self._extract_final_answer(raw_answer),
                    answer_text=raw_answer,
                    metadata=metadata,
                )
            )
            if limit is not None and len(samples) >= limit:
                break

        return samples

    @staticmethod
    def _extract_final_answer(raw_answer: str) -> str:
        marker_match = re.search(r"####\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)", raw_answer)
        if marker_match:
            return marker_match.group(1).replace(",", "")

        number_matches = re.findall(r"[-+]?[0-9][0-9,]*(?:\.[0-9]+)?", raw_answer)
        if number_matches:
            return number_matches[-1].replace(",", "")

        return raw_answer.strip()


def load_dataset(dataset_name: str, path: str | Path | None = None, limit: int | None = None) -> list[DatasetSample]:
    normalized = dataset_name.lower()
    if normalized == "mmlu":
        return MMLULoader().load(path=path, limit=limit)
    if normalized == "gsm8k":
        return GSM8KLoader().load(path=path, limit=limit)
    raise ValueError(f"Unsupported dataset: {dataset_name}")
