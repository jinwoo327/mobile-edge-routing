from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from datasets.common import DatasetSample
from models.ollama_client import OllamaClient


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


def _build_prompt(sample: DatasetSample) -> tuple[str, str]:
    if sample.task_type == "multiple_choice":
        options = "\n".join(f"{chr(ord('A') + index)}. {choice}" for index, choice in enumerate(sample.choices))
        return (
            f"Question:\n{sample.question}\n\nChoices:\n{options}\n\n"
            "Answer with only the single best choice letter.",
            "You are a model for single-turn academic benchmarking. "
            "Be concise and return only the requested answer format.",
        )

    return (
        f"Solve the following math word problem and return only the final numeric answer.\n\nQuestion:\n{sample.question}",
        "You are a model for single-turn academic benchmarking. "
        "Return only the final numeric answer with no explanation.",
    )


def _normalize_prediction(sample: DatasetSample, generated_text: str) -> str:
    if sample.task_type == "multiple_choice":
        match = re.search(r"\b([ABCD])\b", generated_text.upper())
        return match.group(1) if match else generated_text.strip()

    matches = re.findall(r"[-+]?[0-9][0-9,]*(?:\.[0-9]+)?", generated_text)
    if matches:
        return matches[-1].replace(",", "")
    return generated_text.strip()


def _estimate_confidence_from_logprobs(response: dict[str, Any]) -> tuple[float | None, str | None]:
    logprobs = response.get("logprobs") or []
    values = [float(token.get("logprob", 0.0)) for token in logprobs if "logprob" in token]
    if not values:
        return None, None

    average_logprob = sum(values) / len(values)
    return max(0.0, min(1.0, math.exp(average_logprob))), "logprobs"


def _estimate_confidence_from_output(sample: DatasetSample, generated_text: str) -> tuple[float, str]:
    stripped = generated_text.strip()
    if sample.task_type == "multiple_choice":
        if re.fullmatch(r"[ABCDabcd]", stripped):
            return 0.75, "format_heuristic"
        if re.search(r"\b([ABCD])\b", stripped.upper()):
            return 0.6, "format_heuristic"
        return 0.3, "format_heuristic"

    number_matches = re.findall(r"[-+]?[0-9][0-9,]*(?:\.[0-9]+)?", stripped)
    if not number_matches:
        return 0.2, "format_heuristic"
    if stripped == number_matches[-1]:
        return 0.7, "format_heuristic"
    if len(number_matches) == 1:
        return 0.5, "format_heuristic"
    return 0.35, "format_heuristic"


def _extract_confidence(sample: DatasetSample, response: dict[str, Any], generated_text: str) -> tuple[float, str]:
    confidence, source = _estimate_confidence_from_logprobs(response)
    if confidence is not None and source is not None:
        return confidence, source
    return _estimate_confidence_from_output(sample, generated_text)


class MockLocalBackend:
    def __init__(self, model_name: str = "gemma3:1b", quantization: str | None = None):
        self.model_name = model_name
        self.quantization = quantization
        self.name = "mock_local_gemma3_1b"

    def generate(self, sample: DatasetSample) -> InferenceResult:
        prediction = str(sample.metadata.get("mock_local_prediction", self._default_prediction(sample)))
        confidence = float(sample.metadata.get("mock_local_confidence", self._default_confidence(sample)))
        latency_ms = float(sample.metadata.get("mock_local_latency_ms", 35.0))
        return InferenceResult(
            backend_name=self.name,
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata={
                "source": "mock",
                "backend_type": "mock_local_backend",
                "model_name": self.model_name,
                "quantization": self.quantization,
                "confidence_source": "mock_metadata",
            },
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
    def __init__(self, model_name: str = "gemma3:4b", quantization: str | None = None):
        self.model_name = model_name
        self.quantization = quantization
        self.name = "mock_edge_gemma3_4b"

    def generate(self, sample: DatasetSample) -> InferenceResult:
        prediction = str(sample.metadata.get("mock_edge_prediction", self._default_prediction(sample)))
        confidence = float(sample.metadata.get("mock_edge_confidence", 0.92))
        latency_ms = float(sample.metadata.get("mock_edge_latency_ms", 120.0))
        return InferenceResult(
            backend_name=self.name,
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata={
                "source": "mock",
                "backend_type": "mock_edge_backend",
                "model_name": self.model_name,
                "quantization": self.quantization,
                "confidence_source": "mock_metadata",
            },
        )

    def _default_prediction(self, sample: DatasetSample) -> str:
        if sample.task_type == "multiple_choice" and sample.answer:
            return sample.answer
        if sample.answer:
            return sample.answer
        return MockLocalBackend()._default_prediction(sample)


class OllamaGenerationBackend:
    def __init__(
        self,
        backend_type: str,
        backend_name: str,
        base_url: str,
        model_name: str,
        timeout: float,
        quantization: str | None = None,
        client: OllamaClient | None = None,
    ):
        self.backend_type = backend_type
        self.base_url = base_url
        self.model_name = model_name
        self.timeout = timeout
        self.quantization = quantization
        self.client = client or OllamaClient(base_url=base_url, model_name=model_name, timeout=timeout)
        self.name = backend_name

    def health_check(self) -> dict[str, Any]:
        return self.client.health_check()

    def ensure_ready(self) -> dict[str, Any]:
        return self.client.ensure_model_available()

    def generate(self, sample: DatasetSample) -> InferenceResult:
        prompt, system_prompt = _build_prompt(sample)
        started_at = time.perf_counter()
        response = self.client.generate(prompt=prompt, system=system_prompt)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        generated_text = str(response.get("response", "")).strip()
        confidence, confidence_source = _extract_confidence(sample, response, generated_text)
        latency_ms = float(response.get("total_duration", 0)) / 1_000_000 if response.get("total_duration") else elapsed_ms

        return InferenceResult(
            backend_name=self.name,
            prediction=_normalize_prediction(sample, generated_text),
            confidence=confidence,
            latency_ms=latency_ms,
            metadata={
                "source": "ollama",
                "backend_type": self.backend_type,
                "model_name": self.model_name,
                "quantization": self.quantization,
                "base_url": self.base_url,
                "raw_response": generated_text,
                "done_reason": response.get("done_reason"),
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count"),
                "confidence_source": confidence_source,
                "model_response": response,
            },
        )


class RealLocalBackend(OllamaGenerationBackend):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "gemma3:1b",
        timeout: float = 60.0,
        quantization: str | None = None,
        client: OllamaClient | None = None,
    ):
        super().__init__(
            backend_type="real_local_backend",
            backend_name="real_local_backend",
            base_url=base_url,
            model_name=model_name,
            timeout=timeout,
            quantization=quantization,
            client=client,
        )


class OllamaEdgeBackend(OllamaGenerationBackend):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "gemma3:4b",
        timeout: float = 60.0,
        quantization: str | None = None,
        client: OllamaClient | None = None,
    ):
        super().__init__(
            backend_type="ollama_edge_backend",
            backend_name="ollama_edge_backend",
            base_url=base_url,
            model_name=model_name,
            timeout=timeout,
            quantization=quantization,
            client=client,
        )


@dataclass(slots=True)
class BackendConfig:
    backend_type: str
    model_name: str
    quantization: str | None = None
    base_url: str | None = None
    timeout: float | None = None


def create_local_backend(config: BackendConfig) -> ModelBackend:
    if config.backend_type == "mock_local_backend":
        return MockLocalBackend(model_name=config.model_name, quantization=config.quantization)
    if config.backend_type == "real_local_backend":
        return RealLocalBackend(
            base_url=config.base_url or "http://localhost:11434",
            model_name=config.model_name,
            timeout=config.timeout or 60.0,
            quantization=config.quantization,
        )
    raise ValueError(f"Unsupported local backend type: {config.backend_type}")


def create_edge_backend(config: BackendConfig) -> ModelBackend:
    if config.backend_type == "mock_edge_backend":
        return MockEdgeBackend(model_name=config.model_name, quantization=config.quantization)
    if config.backend_type == "ollama_edge_backend":
        return OllamaEdgeBackend(
            base_url=config.base_url or "http://localhost:11434",
            model_name=config.model_name,
            timeout=config.timeout or 60.0,
            quantization=config.quantization,
        )
    raise ValueError(f"Unsupported edge backend type: {config.backend_type}")
