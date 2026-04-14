from __future__ import annotations

import re

from datasets.common import DatasetSample


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _extract_last_number(value: str) -> str:
    matches = re.findall(r"[-+]?[0-9][0-9,]*(?:\.[0-9]+)?", value)
    if not matches:
        return _normalize_text(value)
    return matches[-1].replace(",", "")


def _normalize_multiple_choice(value: str) -> str:
    normalized = _normalize_text(value).upper()
    if normalized in {"A", "B", "C", "D"}:
        return normalized

    match = re.search(r"\b([ABCD])\b", normalized)
    if match:
        return match.group(1)

    return normalized[:1]


def evaluate_prediction(sample: DatasetSample, prediction: str) -> bool:
    if sample.task_type == "multiple_choice":
        return _normalize_multiple_choice(prediction) == _normalize_multiple_choice(sample.answer)
    return _extract_last_number(prediction) == _extract_last_number(sample.answer)
