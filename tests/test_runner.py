import json
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
