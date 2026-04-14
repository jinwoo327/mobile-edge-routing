import unittest

from datasets.common import DatasetSample
from evaluation.metrics import evaluate_prediction
from evaluation.summary import summarize_results


class MetricsTests(unittest.TestCase):
    def test_multiple_choice_metric_matches_letters(self) -> None:
        sample = DatasetSample(
            sample_id="1",
            dataset="mmlu",
            task_type="multiple_choice",
            question="Question",
            choices=["A1", "A2", "A3", "A4"],
            answer="C",
        )
        self.assertTrue(evaluate_prediction(sample, "c"))

    def test_gsm8k_metric_extracts_numbers(self) -> None:
        sample = DatasetSample(
            sample_id="2",
            dataset="gsm8k",
            task_type="free_form",
            question="Question",
            answer="24",
        )
        self.assertTrue(evaluate_prediction(sample, "The answer is 24."))

    def test_summary_groups_partial_blocks(self) -> None:
        rows = summarize_results(
            [
                {
                    "dataset": "mmlu",
                    "mode": "hybrid_only",
                    "is_correct": True,
                    "fell_back_to_edge": False,
                    "latency_ms": 30.0,
                    "backend_used": "mock_local_gemma3_1b",
                },
                {
                    "dataset": "mmlu",
                    "mode": "hybrid_only",
                    "is_correct": False,
                    "fell_back_to_edge": True,
                    "latency_ms": 120.0,
                    "backend_used": "mock_edge_gemma3_4b",
                },
            ],
            block_sizes=[20],
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sample_count"], 2)
        self.assertEqual(rows[0]["fallback_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
