import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from runner.experiment import run_experiment


class RunnerTests(unittest.TestCase):
    def test_run_experiment_writes_quantization_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = run_experiment(
                dataset_name="mmlu",
                mode="hybrid_only",
                limit=1,
                log_dir=tmp_dir,
                local_quantization="Q4_K_M",
                edge_quantization="Q4_K_M",
            )

            metadata = json.loads(Path(artifacts.metadata_path).read_text(encoding="utf-8"))
            self.assertEqual(metadata["local_backend"]["quantization"], "Q4_K_M")
            self.assertEqual(metadata["edge_backend"]["quantization"], "Q4_K_M")

            record = json.loads(Path(artifacts.jsonl_path).read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["run_metadata"]["local_backend"]["quantization"], "Q4_K_M")
            self.assertEqual(record["run_metadata"]["edge_backend"]["quantization"], "Q4_K_M")
            self.assertEqual(record["local_quantization"], "Q4_K_M")
            self.assertEqual(record["edge_quantization"], "Q4_K_M")
            self.assertEqual(record["dataset_source"], "fixture")
            self.assertEqual(record["local_confidence_estimator"], "router_v1")

    def test_run_experiment_writes_backend_identity_to_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = run_experiment(
                dataset_name="mmlu",
                mode="hybrid_only",
                limit=1,
                log_dir=tmp_dir,
                local_backend_type="mock_local_backend",
                local_model_name="gemma3:1b",
                edge_backend_type="mock_edge_backend",
                edge_model_name="gemma3:4b",
            )

            summary_lines = Path(artifacts.summary_csv_path).read_text(encoding="utf-8").splitlines()
            self.assertIn("local_backend_type", summary_lines[0])
            self.assertIn("edge_quantization", summary_lines[0])
            self.assertIn("mock_local_backend", summary_lines[1])
            self.assertIn("mock_edge_backend", summary_lines[1])

    def test_run_experiment_prints_dataset_limit_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = io.StringIO()
            with redirect_stdout(output):
                artifacts = run_experiment(
                    dataset_name="gsm8k",
                    dataset_source="fixture",
                    mode="local_only",
                    limit=20,
                    log_dir=tmp_dir,
                )

            self.assertEqual(artifacts.sample_count, 4)
            self.assertIn("Requested limit 20", output.getvalue())


if __name__ == "__main__":
    unittest.main()
