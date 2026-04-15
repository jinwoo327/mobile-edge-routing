import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from runner.experiment import run_experiment


class RunnerTests(unittest.TestCase):
    def test_run_experiment_writes_quantization_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = run_experiment(
                dataset_name="mmlu",
                mode="hybrid_only",
                limit=1,
                log_dir=tmp_dir,
                prompt_profile="benchmark_v2",
                local_quantization="Q4_K_M",
                local_temperature=0.0,
                edge_quantization="Q4_K_M",
                edge_temperature=0.0,
            )

            metadata = json.loads(Path(artifacts.metadata_path).read_text(encoding="utf-8"))
            self.assertEqual(metadata["local_backend"]["quantization"], "Q4_K_M")
            self.assertEqual(metadata["edge_backend"]["quantization"], "Q4_K_M")
            self.assertEqual(metadata["prompt_profile"], "benchmark_v2")
            self.assertEqual(metadata["local_backend"]["temperature"], 0.0)
            self.assertEqual(metadata["edge_backend"]["temperature"], 0.0)

            record = json.loads(Path(artifacts.jsonl_path).read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["run_metadata"]["local_backend"]["quantization"], "Q4_K_M")
            self.assertEqual(record["run_metadata"]["edge_backend"]["quantization"], "Q4_K_M")
            self.assertEqual(record["local_quantization"], "Q4_K_M")
            self.assertEqual(record["edge_quantization"], "Q4_K_M")
            self.assertEqual(record["prompt_profile"], "benchmark_v2")
            self.assertEqual(record["local_temperature"], 0.0)
            self.assertEqual(record["edge_temperature"], 0.0)
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

    def test_hybrid_latency_is_logged_as_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = run_experiment(
                dataset_name="gsm8k",
                dataset_source="fixture",
                mode="hybrid_only",
                limit=1,
                confidence_threshold=0.8,
                log_dir=tmp_dir,
                local_backend_type="mock_local_backend",
                edge_backend_type="mock_edge_backend",
            )

            record = json.loads(Path(artifacts.jsonl_path).read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(record["local_latency_ms"], 35.0)
            self.assertEqual(record["edge_latency_ms"], 120.0)
            self.assertEqual(record["final_backend_latency_ms"], 120.0)
            self.assertEqual(record["end_to_end_latency_ms"], 155.0)
            self.assertEqual(record["latency_ms"], 155.0)

    def test_run_experiment_writes_error_json_on_sample_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            call_count = {"value": 0}

            def failing_execute_mode(*args, **kwargs):
                call_count["value"] += 1
                if call_count["value"] == 2:
                    raise RuntimeError("synthetic failure")
                sample = kwargs["sample"]
                backend = kwargs["local_backend"]
                result = backend.generate(sample)
                from routing.modes import RouteDecision

                return RouteDecision(
                    sample_id=sample.sample_id,
                    mode="local_only",
                    backend_used=result.backend_name,
                    final_result=result,
                    local_result=result,
                )

            with patch("runner.experiment.execute_mode", side_effect=failing_execute_mode):
                with self.assertRaises(RuntimeError):
                    run_experiment(
                        dataset_name="gsm8k",
                        dataset_source="fixture",
                        mode="local_only",
                        limit=2,
                        log_dir=tmp_dir,
                        local_backend_type="mock_local_backend",
                        edge_backend_type="mock_edge_backend",
                    )

            error_files = list(Path(tmp_dir).glob("run-*-error.json"))
            self.assertEqual(len(error_files), 1)
            payload = json.loads(error_files[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["failed_sample_index"], 2)
            self.assertEqual(payload["exception_type"], "RuntimeError")
            self.assertIn("synthetic failure", payload["exception_message"])

    def test_run_experiment_can_skip_failed_samples_and_fill_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            failing_ids = {"gsm8k-2"}

            def flaky_execute_mode(*args, **kwargs):
                sample = kwargs["sample"]
                if sample.sample_id in failing_ids:
                    raise RuntimeError("synthetic failure")
                backend = kwargs["local_backend"]
                result = backend.generate(sample)
                from routing.modes import RouteDecision

                return RouteDecision(
                    sample_id=sample.sample_id,
                    mode="local_only",
                    backend_used=result.backend_name,
                    final_result=result,
                    local_result=result,
                )

            with patch("runner.experiment.execute_mode", side_effect=flaky_execute_mode):
                artifacts = run_experiment(
                    dataset_name="gsm8k",
                    dataset_source="fixture",
                    mode="local_only",
                    limit=3,
                    skip_failed_samples=True,
                    log_dir=tmp_dir,
                    local_backend_type="mock_local_backend",
                    edge_backend_type="mock_edge_backend",
                )

            rows = [
                json.loads(line)
                for line in Path(artifacts.jsonl_path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(artifacts.sample_count, 3)
            self.assertEqual(len(rows), 3)
            self.assertEqual([row["sample_id"] for row in rows], ["gsm8k-1", "gsm8k-3", "gsm8k-4"])
            error_logs = list(Path(tmp_dir).glob("run-*-errors.jsonl"))
            self.assertEqual(len(error_logs), 1)
            error_entries = error_logs[0].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(error_entries), 1)
            self.assertIn("gsm8k-2", error_entries[0])


if __name__ == "__main__":
    unittest.main()
