from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from datasets.common import DatasetSample
from models.backends import InferenceResult, ModelBackend


ExecutionMode = Literal["local_only", "edge_only", "hybrid_only"]


@dataclass(slots=True)
class RouteDecision:
    sample_id: str
    mode: ExecutionMode
    backend_used: str
    final_result: InferenceResult
    local_result: InferenceResult | None = None
    edge_result: InferenceResult | None = None
    fell_back_to_edge: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class HybridRouter:
    def __init__(self, local_backend: ModelBackend, edge_backend: ModelBackend, confidence_threshold: float = 0.6):
        self.local_backend = local_backend
        self.edge_backend = edge_backend
        self.confidence_threshold = confidence_threshold

    def run(self, sample: DatasetSample) -> RouteDecision:
        local_result = self.local_backend.generate(sample)
        fell_back = local_result.confidence < self.confidence_threshold
        if fell_back:
            edge_result = self.edge_backend.generate(sample)
            final_result = edge_result
            backend_used = edge_result.backend_name
        else:
            edge_result = None
            final_result = local_result
            backend_used = local_result.backend_name

        return RouteDecision(
            sample_id=sample.sample_id,
            mode="hybrid_only",
            backend_used=backend_used,
            final_result=final_result,
            local_result=local_result,
            edge_result=edge_result,
            fell_back_to_edge=fell_back,
            metadata={"confidence_threshold": self.confidence_threshold},
        )


def execute_mode(
    sample: DatasetSample,
    mode: ExecutionMode,
    local_backend: ModelBackend,
    edge_backend: ModelBackend,
    confidence_threshold: float = 0.6,
) -> RouteDecision:
    if mode == "local_only":
        result = local_backend.generate(sample)
        return RouteDecision(
            sample_id=sample.sample_id,
            mode=mode,
            backend_used=result.backend_name,
            final_result=result,
            local_result=result,
        )

    if mode == "edge_only":
        result = edge_backend.generate(sample)
        return RouteDecision(
            sample_id=sample.sample_id,
            mode=mode,
            backend_used=result.backend_name,
            final_result=result,
            edge_result=result,
        )

    if mode == "hybrid_only":
        return HybridRouter(
            local_backend=local_backend,
            edge_backend=edge_backend,
            confidence_threshold=confidence_threshold,
        ).run(sample)

    raise ValueError(f"Unsupported execution mode: {mode}")
