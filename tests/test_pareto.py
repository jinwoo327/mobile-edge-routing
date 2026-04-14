import tempfile
import unittest
from pathlib import Path

from evaluation.pareto import pareto_frontier
from runner.sweep import summarize_jsonl_run, write_sweep_csv


class ParetoTests(unittest.TestCase):
    def test_pareto_frontier_removes_dominated_points(self) -> None:
        rows = [
            {"threshold": 0.4, "accuracy": 0.8, "mean_latency_ms": 300.0, "edge_rate": 0.6},
            {"threshold": 0.6, "accuracy": 0.8, "mean_latency_ms": 250.0, "edge_rate": 0.5},
            {"threshold": 0.8, "accuracy": 0.7, "mean_latency_ms": 400.0, "edge_rate": 0.8},
        ]
        frontier = pareto_frontier(rows, maximize=["accuracy"], minimize=["mean_latency_ms", "edge_rate"])
        self.assertEqual(len(frontier), 1)
        self.assertEqual(frontier[0]["threshold"], 0.6)

    def test_summarize_jsonl_run_extracts_tradeoff_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "run.jsonl"
            jsonl_path.write_text(
                "\n".join(
                    [
                        '{"dataset":"gsm8k","dataset_source":"real","mode":"hybrid_only","backend_role":"local","fell_back_to_edge":false,"is_correct":true,"latency_ms":40.0,"local_confidence":0.7,"local_confidence_source":"router_v1","local_backend_type":"real_local_backend","local_model_name":"gemma3:1b","local_quantization":null,"edge_backend_type":"ollama_edge_backend","edge_model_name":"gemma3:4b","edge_quantization":null}',
                        '{"dataset":"gsm8k","dataset_source":"real","mode":"hybrid_only","backend_role":"edge","fell_back_to_edge":true,"is_correct":false,"latency_ms":120.0,"local_confidence":0.4,"local_confidence_source":"router_v1","local_backend_type":"real_local_backend","local_model_name":"gemma3:1b","local_quantization":null,"edge_backend_type":"ollama_edge_backend","edge_model_name":"gemma3:4b","edge_quantization":null}',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            summary = summarize_jsonl_run(jsonl_path, exclude_first_n_latency=1)

        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["edge_rate"], 0.5)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["local_confidence_source"], "router_v1")
        self.assertEqual(summary["latency_exclude_first_n"], 1)
        self.assertEqual(summary["mean_latency_ms_excluding_warmup"], 120.0)

    def test_write_sweep_csv_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "sweep.csv"
            write_sweep_csv(output_path, [{"threshold": 0.6, "accuracy": 0.8, "mean_latency_ms": 100.0}])
            self.assertTrue(output_path.exists())
            self.assertIn("threshold", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
