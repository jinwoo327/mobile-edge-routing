import tempfile
import unittest
from pathlib import Path

from datasets.loaders import GSM8KLoader, MMLULoader


class DatasetLoaderTests(unittest.TestCase):
    def test_mmlu_loader_normalizes_answer_letter(self) -> None:
        samples = MMLULoader().load(limit=1)
        sample = samples[0]
        self.assertEqual(sample.dataset, "mmlu")
        self.assertEqual(sample.task_type, "multiple_choice")
        self.assertEqual(sample.answer, "B")
        self.assertEqual(sample.answer_text, "Paris")

    def test_gsm8k_loader_extracts_final_numeric_answer(self) -> None:
        samples = GSM8KLoader().load(limit=1)
        sample = samples[0]
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
            sample = MMLULoader().load(path=input_path, limit=1)[0]
            self.assertEqual(sample.answer, "B")
            self.assertEqual(sample.answer_text, "4")


if __name__ == "__main__":
    unittest.main()
