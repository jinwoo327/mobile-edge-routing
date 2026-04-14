import tempfile
import unittest
from pathlib import Path

from evaluation.compare import compare_runs, write_comparison_csv


class CompareTests(unittest.TestCase):
    def test_compare_runs_orders_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            payloads = {
                "run-local.jsonl": '{"dataset":"mmlu","dataset_source":"real","mode":"local_only","backend_role":"local","fell_back_to_edge":false,"is_correct":true,"latency_ms":40.0,"local_confidence":0.7,"local_confidence_source":"router_v1","local_backend_type":"real_local_backend","local_model_name":"gemma3:1b","local_quantization":null,"edge_backend_type":"ollama_edge_backend","edge_model_name":"gemma3:4b","edge_quantization":null}\n',
                "run-hybrid.jsonl": '{"dataset":"mmlu","dataset_source":"real","mode":"hybrid_only","backend_role":"edge","fell_back_to_edge":true,"is_correct":true,"latency_ms":100.0,"local_confidence":0.3,"local_confidence_source":"router_v1","local_backend_type":"real_local_backend","local_model_name":"gemma3:1b","local_quantization":null,"edge_backend_type":"ollama_edge_backend","edge_model_name":"gemma3:4b","edge_quantization":null}\n',
                "run-edge.jsonl": '{"dataset":"mmlu","dataset_source":"real","mode":"edge_only","backend_role":"edge","fell_back_to_edge":false,"is_correct":true,"latency_ms":120.0,"local_confidence":null,"local_confidence_source":null,"local_backend_type":"real_local_backend","local_model_name":"gemma3:1b","local_quantization":null,"edge_backend_type":"ollama_edge_backend","edge_model_name":"gemma3:4b","edge_quantization":null}\n',
            }

            paths = []
            for name, content in payloads.items():
                path = root / name
                path.write_text(content, encoding="utf-8")
                paths.append(path)

            rows = compare_runs(paths, exclude_first_n_latency=1)

        self.assertEqual([row["mode"] for row in rows], ["local_only", "hybrid_only", "edge_only"])
        self.assertEqual(rows[0]["latency_exclude_first_n"], 1)

    def test_write_comparison_csv_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "comparison.csv"
            write_comparison_csv(output_path, [{"mode": "local_only", "accuracy": 0.5}])
            self.assertTrue(output_path.exists())
            self.assertIn("mode", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
