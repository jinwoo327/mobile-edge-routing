import json
import unittest
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from models.backends import BackendConfig, OllamaEdgeBackend, create_edge_backend
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


if __name__ == "__main__":
    unittest.main()
