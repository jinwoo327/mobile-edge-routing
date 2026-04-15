import tempfile
import unittest
from pathlib import Path

from evaluation.presentation import build_presentation_rows, write_presentation_csv, write_presentation_json


class PresentationTests(unittest.TestCase):
    def test_build_presentation_rows_extracts_core_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            compare_csv = Path(tmp_dir) / "compare.csv"
            compare_csv.write_text(
                "\n".join(
                    [
                        "sample_count,accuracy,median_latency_ms,median_latency_ms_excluding_warmup,p25_latency_ms_excluding_warmup,p75_latency_ms_excluding_warmup,edge_rate,fallback_rate,dataset,dataset_source,mode,run_id,confidence_threshold,local_confidence_estimator",
                        "100,0.36,423.87,423.72,405.88,437.92,0.48,0.48,mmlu,real,hybrid_only,run-123,0.6,router_v1",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = build_presentation_rows(compare_csv)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["mode"], "hybrid_only")
            self.assertEqual(rows[0]["sample_count"], 100)
            self.assertEqual(rows[0]["latency_iqr_ms_excluding_warmup"], "405.88-437.92")

            output_csv = Path(tmp_dir) / "presentation.csv"
            output_json = Path(tmp_dir) / "presentation.json"
            write_presentation_csv(output_csv, rows)
            write_presentation_json(output_json, rows)

            self.assertTrue(output_csv.exists())
            self.assertTrue(output_json.exists())


if __name__ == "__main__":
    unittest.main()
