import json
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from datasets.common import DatasetSample
from models.backends import BackendConfig, OllamaEdgeBackend, RealLocalBackend, create_edge_backend, create_local_backend
from models.ollama_client import OllamaClient, OllamaConnectionError, OllamaModelNotFoundError


class FakeHttpResponse:
    def __init__(self, payload: dict):
        self.payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "FakeHttpResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class FakeOllamaClient:
    def __init__(self, response: dict):
        self.response = response
        self.last_prompt = None
        self.last_system = None
        self.last_options = None

    def health_check(self) -> dict:
        return {"reachable": True, "model_available": True}

    def ensure_model_available(self) -> dict:
        return self.health_check()

    def generate(self, prompt: str, system: str | None = None, options: dict | None = None) -> dict:
        self.last_prompt = prompt
        self.last_system = system
        self.last_options = options
        return self.response


class OllamaTests(unittest.TestCase):
    def test_health_check_reports_model_details(self) -> None:
        client = OllamaClient(base_url="http://localhost:11434", model_name="gemma3:4b", timeout=5.0)
        payload = {
            "models": [
                {
                    "name": "gemma3:4b",
                    "details": {"quantization_level": "Q4_K_M", "parameter_size": "4.3B"},
                }
            ]
        }
        with patch("models.ollama_client.request.urlopen", return_value=FakeHttpResponse(payload)):
            health = client.health_check()

        self.assertTrue(health["reachable"])
        self.assertTrue(health["model_available"])
        self.assertEqual(health["model_details"]["quantization_level"], "Q4_K_M")

    def test_generate_raises_model_not_found(self) -> None:
        client = OllamaClient(base_url="http://localhost:11434", model_name="missing-model", timeout=5.0)
        http_error = HTTPError(
            url="http://localhost:11434/api/generate",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=BytesIO(b'{"error":"model \\"missing-model\\" not found, try pulling it first"}'),
        )
        with patch("models.ollama_client.request.urlopen", side_effect=http_error):
            with self.assertRaises(OllamaModelNotFoundError):
                client.generate(prompt="hello")

    def test_generate_raises_connection_error(self) -> None:
        client = OllamaClient(base_url="http://localhost:11434", model_name="gemma3:4b", timeout=5.0)
        with patch("models.ollama_client.request.urlopen", side_effect=URLError("connection refused")):
            with self.assertRaises(OllamaConnectionError):
                client.health_check()

    def test_backend_factory_creates_ollama_backend(self) -> None:
        backend = create_edge_backend(
            BackendConfig(
                backend_type="ollama_edge_backend",
                model_name="gemma3:4b",
                base_url="http://localhost:11434",
                timeout=60.0,
                quantization="Q4_K_M",
            )
        )
        self.assertIsInstance(backend, OllamaEdgeBackend)
        self.assertEqual(backend.quantization, "Q4_K_M")

    def test_backend_factory_creates_real_local_backend(self) -> None:
        backend = create_local_backend(
            BackendConfig(
                backend_type="real_local_backend",
                model_name="gemma3:1b",
                base_url="http://localhost:11434",
                timeout=60.0,
                quantization="Q4_K_M",
            )
        )
        self.assertIsInstance(backend, RealLocalBackend)
        self.assertEqual(backend.quantization, "Q4_K_M")

    def test_real_local_backend_uses_fallback_confidence_estimator(self) -> None:
        sample = DatasetSample(
            sample_id="1",
            dataset="mmlu",
            task_type="multiple_choice",
            question="Capital of France?",
            choices=["Berlin", "Paris", "Rome", "Madrid"],
            answer="B",
        )
        backend = RealLocalBackend(
            client=FakeOllamaClient({"response": "B", "total_duration": 1000000}),
            model_name="gemma3:1b",
        )
        result = backend.generate(sample)

        self.assertEqual(result.prediction, "B")
        self.assertEqual(result.metadata["confidence_source"], "format_heuristic")
        self.assertGreater(result.confidence, 0.0)

    def test_real_local_backend_passes_temperature_and_prompt_profile(self) -> None:
        sample = DatasetSample(
            sample_id="1",
            dataset="mmlu",
            task_type="multiple_choice",
            question="Capital of France?",
            choices=["Berlin", "Paris", "Rome", "Madrid"],
            answer="B",
        )
        client = FakeOllamaClient({"response": "B", "total_duration": 1000000})
        backend = RealLocalBackend(
            client=client,
            model_name="gemma3:1b",
            temperature=0.0,
            prompt_profile="benchmark_v2",
        )

        result = backend.generate(sample)

        self.assertEqual(client.last_options, {"temperature": 0.0})
        self.assertIn("Return exactly one uppercase letter", client.last_prompt)
        self.assertIn("Follow the examples", client.last_prompt)
        self.assertGreaterEqual(client.last_prompt.count("Answer:"), 5)
        self.assertEqual(result.metadata["temperature"], 0.0)
        self.assertEqual(result.metadata["prompt_profile"], "benchmark_v2")

    def test_gsm8k_benchmark_v2_uses_answer_line_format(self) -> None:
        sample = DatasetSample(
            sample_id="1",
            dataset="gsm8k",
            task_type="open_ended",
            question="Tom has 3 apples and buys 4 more. How many apples does he have now?",
            answer="7",
        )
        client = FakeOllamaClient({"response": "Add 3 and 4.\nAnswer: 7", "total_duration": 1000000})
        backend = RealLocalBackend(
            client=client,
            model_name="gemma3:1b",
            temperature=0.0,
            prompt_profile="benchmark_v2",
        )

        result = backend.generate(sample)

        self.assertIn("Solve the problem step by step briefly.", client.last_prompt)
        self.assertIn("On the last line, output exactly: Answer: <number>", client.last_prompt)
        self.assertEqual(result.prediction, "7")

    def test_router_v1_penalizes_more_complex_question(self) -> None:
        simple_sample = DatasetSample(
            sample_id="1",
            dataset="mmlu",
            task_type="multiple_choice",
            question="Capital of France?",
            choices=["Berlin", "Paris", "Rome", "Madrid"],
            answer="B",
        )
        complex_sample = DatasetSample(
            sample_id="2",
            dataset="mmlu",
            task_type="multiple_choice",
            question=(
                "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q "
                "and identify the correct option among the following statements."
            ),
            choices=["1", "2", "4", "8"],
            answer="C",
        )
        backend = RealLocalBackend(
            client=FakeOllamaClient({"response": "C", "total_duration": 1000000}),
            model_name="gemma3:1b",
            confidence_estimator="router_v1",
        )

        simple_result = backend.generate(simple_sample)
        complex_result = backend.generate(complex_sample)

        self.assertEqual(simple_result.metadata["confidence_source"], "router_v1")
        self.assertLess(complex_result.confidence, simple_result.confidence)

    def test_backend_factory_passes_prompt_profile_and_temperature(self) -> None:
        backend = create_local_backend(
            BackendConfig(
                backend_type="real_local_backend",
                model_name="gemma3:1b",
                base_url="http://localhost:11434",
                timeout=60.0,
                quantization="Q4_K_M",
                temperature=0.0,
                prompt_profile="benchmark_v2",
                confidence_estimator="router_v1",
            )
        )
        self.assertIsInstance(backend, RealLocalBackend)
        self.assertEqual(backend.temperature, 0.0)
        self.assertEqual(backend.prompt_profile, "benchmark_v2")


if __name__ == "__main__":
    unittest.main()
