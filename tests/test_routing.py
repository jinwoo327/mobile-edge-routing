import unittest

from datasets.common import DatasetSample
from models.backends import MockEdgeBackend, MockLocalBackend
from routing.modes import execute_mode


class RoutingTests(unittest.TestCase):
    def test_hybrid_falls_back_when_local_confidence_is_low(self) -> None:
        sample = DatasetSample(
            sample_id="1",
            dataset="gsm8k",
            task_type="free_form",
            question="1 + 1 = ?",
            answer="2",
            metadata={
                "mock_local_prediction": "3",
                "mock_local_confidence": 0.2,
                "mock_edge_prediction": "2",
                "mock_edge_confidence": 0.9,
            },
        )
        decision = execute_mode(sample, "hybrid_only", MockLocalBackend(), MockEdgeBackend(), confidence_threshold=0.6)
        self.assertTrue(decision.fell_back_to_edge)
        self.assertEqual(decision.backend_used, "mock_edge_gemma3_4b")
        self.assertEqual(decision.final_result.prediction, "2")

    def test_hybrid_keeps_local_result_when_confident(self) -> None:
        sample = DatasetSample(
            sample_id="2",
            dataset="mmlu",
            task_type="multiple_choice",
            question="Capital of France?",
            choices=["Berlin", "Paris", "Rome", "Madrid"],
            answer="B",
            metadata={"mock_local_prediction": "B", "mock_local_confidence": 0.8},
        )
        decision = execute_mode(sample, "hybrid_only", MockLocalBackend(), MockEdgeBackend(), confidence_threshold=0.6)
        self.assertFalse(decision.fell_back_to_edge)
        self.assertEqual(decision.backend_used, "mock_local_gemma3_1b")
        self.assertEqual(decision.final_result.prediction, "B")


if __name__ == "__main__":
    unittest.main()
