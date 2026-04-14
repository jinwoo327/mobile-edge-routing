from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from datasets.common import DatasetSample


@dataclass(slots=True)
class InferenceResult:
    backend_name: str
    prediction: str
    confidence: float
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelBackend(Protocol):
    name: str

    def generate(self, sample: DatasetSample) -> InferenceResult:
        ...


class MockLocalBackend:
    name = "mock_local_gemma3_1b"

    def generate(self, sample: DatasetSample) -> InferenceResult:
        prediction = str(sample.metadata.get("mock_local_prediction", self._default_prediction(sample)))
        confidence = float(sample.metadata.get("mock_local_confidence", self._default_confidence(sample)))
        latency_ms = float(sample.metadata.get("mock_local_latency_ms", 35.0))
        return InferenceResult(
            backend_name=self.name,
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata={"source": "mock"},
        )

    @staticmethod
    def _default_confidence(sample: DatasetSample) -> float:
        return 0.7 if sample.task_type == "multiple_choice" else 0.45

    def _default_prediction(self, sample: DatasetSample) -> str:
        if sample.task_type == "multiple_choice":
            return "A"

        numbers = [int(match) for match in re.findall(r"\d+", sample.question)]
        if len(numbers) >= 2:
            return str(numbers[0] + numbers[1])
        if numbers:
            return str(numbers[0])
        return "0"


class MockEdgeBackend:
    name = "mock_edge_gemma3_4b"

    def generate(self, sample: DatasetSample) -> InferenceResult:
        prediction = str(sample.metadata.get("mock_edge_prediction", self._default_prediction(sample)))
        confidence = float(sample.metadata.get("mock_edge_confidence", 0.92))
        latency_ms = float(sample.metadata.get("mock_edge_latency_ms", 120.0))
        return InferenceResult(
            backend_name=self.name,
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata={"source": "mock"},
        )

    def _default_prediction(self, sample: DatasetSample) -> str:
        if sample.task_type == "multiple_choice" and sample.answer:
            return sample.answer
        if sample.answer:
            return sample.answer
        return MockLocalBackend()._default_prediction(sample)
