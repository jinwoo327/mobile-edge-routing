from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any
from urllib import error, request


class OllamaError(RuntimeError):
    """Base error for Ollama client failures."""


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama service cannot be reached."""


class OllamaModelNotFoundError(OllamaError):
    """Raised when the requested Ollama model is not available."""


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an unexpected error response."""


@dataclass(slots=True)
class OllamaClient:
    base_url: str = "http://localhost:11434"
    model_name: str = "gemma3:4b"
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    def health_check(self) -> dict[str, Any]:
        tags = self._request_json("/api/tags", method="GET")
        models = tags.get("models", [])
        matched_model = next(
            (
                model
                for model in models
                if model.get("name") == self.model_name or model.get("model") == self.model_name
            ),
            None,
        )
        is_available = matched_model is not None

        response: dict[str, Any] = {
            "reachable": True,
            "base_url": self.base_url,
            "model_name": self.model_name,
            "model_available": is_available,
            "installed_models": [model.get("name", model.get("model")) for model in models],
        }

        if matched_model:
            response["model_details"] = matched_model.get("details", {})

        return response

    def ensure_model_available(self) -> dict[str, Any]:
        health = self.health_check()
        if not health["model_available"]:
            raise OllamaModelNotFoundError(
                f"Model '{self.model_name}' was not found at {self.base_url}. "
                f"Pull it first with: ollama pull {self.model_name}"
            )
        return health

    def generate(self, prompt: str, system: str | None = None, options: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options
        return self._request_json("/api/generate", method="POST", payload=payload)

    def _request_json(self, path: str, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"

        req = request.Request(url, data=data, headers=headers, method=method)

        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw_body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            details = self._parse_error_body(body)
            message = details.get("error", body.strip() or exc.reason)
            if self._is_model_not_found(exc.code, message):
                raise OllamaModelNotFoundError(
                    f"Model '{self.model_name}' was not found at {self.base_url}. "
                    f"Ollama said: {message}"
                ) from exc
            raise OllamaResponseError(f"Ollama request failed with HTTP {exc.code}: {message}") from exc
        except (error.URLError, socket.timeout, TimeoutError, ConnectionError) as exc:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Make sure the service is running and reachable."
            ) from exc

        try:
            return json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise OllamaResponseError(f"Ollama returned invalid JSON from {url}.") from exc

    @staticmethod
    def _parse_error_body(body: str) -> dict[str, Any]:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            return {"error": body.strip()}
        if isinstance(parsed, dict):
            return parsed
        return {"error": body.strip()}

    @staticmethod
    def _is_model_not_found(status_code: int, message: str) -> bool:
        normalized = message.lower()
        return status_code == 404 or "not found" in normalized or "pull" in normalized
