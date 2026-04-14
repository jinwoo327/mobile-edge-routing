from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


TaskType = Literal["multiple_choice", "free_form"]


@dataclass(slots=True)
class DatasetSample:
    sample_id: str
    dataset: str
    task_type: TaskType
    question: str
    choices: list[str] = field(default_factory=list)
    answer: str = ""
    answer_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
