import tempfile
import unittest
from pathlib import Path

from datasets.loaders import GSM8KLoader, MMLULoader, load_dataset


class DatasetLoaderTests(unittest.TestCase):
    def test_mmlu_loader_normalizes_answer_letter(self) -> None:
        result = MMLULoader().load(limit=1)
        sample = result.samples[0]
        self.assertEqual(sample.dataset, "mmlu")
        self.assertEqual(sample.task_type, "multiple_choice")
        self.assertEqual(sample.answer, "B")
        self.assertEqual(sample.answer_text, "Paris")
        self.assertEqual(result.dataset_source, "fixture")

    def test_gsm8k_loader_extracts_final_numeric_answer(self) -> None:
        result = GSM8KLoader().load(limit=1)
        sample = result.samples[0]
        self.assertEqual(sample.dataset, "gsm8k")
        self.assertEqual(sample.task_type, "free_form")
        self.assertEqual(sample.answer, "5")

    def test_loader_accepts_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "mmlu.jsonl"
            input_path.write_text(
                '{"id":"1","question":"2+2?","choices":["3","4","5","6"],"answer":"B"}\n',
                encoding="utf-8",
            )
            sample = MMLULoader().load(path=input_path, limit=1).samples[0]
            self.assertEqual(sample.answer, "B")
            self.assertEqual(sample.answer_text, "4")

    def test_real_gsm8k_loader_supports_directory_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "gsm8k"
            data_dir.mkdir(parents=True)
            (data_dir / "test.jsonl").write_text(
                '{"id":"g1","question":"1+2?","answer":"#### 3"}\n'
                '{"id":"g2","question":"2+2?","answer":"#### 4"}\n',
                encoding="utf-8",
            )
            result = load_dataset("gsm8k", path=data_dir, limit=2, dataset_source="real")

        self.assertEqual(result.dataset_source, "real")
        self.assertEqual(result.available_count, 2)
        self.assertEqual(len(result.samples), 2)
        self.assertEqual(result.samples[0].answer, "3")

    def test_real_mmlu_loader_supports_csv_directory_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir) / "mmlu"
            data_dir.mkdir(parents=True)
            (data_dir / "abstract_algebra_test.csv").write_text(
                "question,A,B,C,D,answer\n"
                "2+2=?,3,4,5,6,B\n"
                "Capital of France?,Berlin,Paris,Rome,Madrid,B\n",
                encoding="utf-8",
            )
            result = load_dataset("mmlu", path=data_dir, limit=2, dataset_source="real")

        self.assertEqual(result.dataset_source, "real")
        self.assertEqual(result.available_count, 2)
        self.assertEqual(result.samples[0].metadata["subject"], "abstract_algebra_test")
        self.assertEqual(result.samples[0].answer, "B")

    def test_limit_warning_is_emitted_when_dataset_is_smaller_than_requested(self) -> None:
        result = load_dataset("gsm8k", limit=20, dataset_source="fixture")
        self.assertEqual(len(result.samples), 4)
        self.assertIn("Requested limit 20", result.warning)


if __name__ == "__main__":
    unittest.main()
