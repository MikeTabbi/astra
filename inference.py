"""Decoupled inference layer for Project ASTRA (Stage 3).

Local backend targets the host MacBook's Ollama node over the LAN.
Cloud backend (Bedrock) drops in at the same interface in Weeks 5-6.
"""

import logging
import os
from typing import Any, Optional, Protocol

from ollama import Client

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")


class InferenceProvider(Protocol):
    """Backend-agnostic inference contract (spec Stage 3)."""

    def complete_json(
        self, prompt: str, temperature: float = 0.2, schema: Optional[dict] = None
    ) -> str:
        """Return raw JSON text from the model. Caller validates with Pydantic."""
        ...


class OllamaProvider:
    """Local backend: 4-bit quantized model on the host MacBook's M2 GPU."""

    def __init__(
        self,
        host: str = DEFAULT_OLLAMA_HOST,
        model: str = DEFAULT_MODEL,
        verify: bool = True,
    ) -> None:
        self.host = host
        self.model = model
        self._client = Client(host=host)
        if verify:
            self._verify()

    def _verify(self) -> None:
        try:
            self._client.list()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}. Confirm the host MacBook is running "
                f"'export OLLAMA_HOST=0.0.0.0 && ollama serve', that you are on the lab LAN, "
                f"and that port 11434 is reachable (try: ping the host, then curl {self.host})."
            ) from exc
        logger.info("Connected to Ollama node at %s (model=%s)", self.host, self.model)

    def complete_json(
        self, prompt: str, temperature: float = 0.2, schema: Optional[dict] = None
    ) -> str:
        response: Any = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=schema if schema is not None else "json",
            options={"temperature": temperature},
        )
        try:
            return response["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response shape from Ollama: {response!r}") from exc


class BedrockProvider:
    """Cloud backend placeholder (Weeks 5-6). Same interface, zero downstream changes."""

    def __init__(self, model_id: str, region: str = "us-east-1") -> None:
        self.model_id = model_id
        self.region = region

    def complete_json(
        self, prompt: str, temperature: float = 0.2, schema: Optional[dict] = None
    ) -> str:
        raise NotImplementedError("Bedrock backend lands in the Weeks 5-6 migration.")


def get_provider(backend: Optional[str] = None) -> InferenceProvider:
    """Single config switch between local and cloud engines (spec Stage 3)."""
    backend = (backend or os.getenv("ASTRA_BACKEND", "local")).lower()
    if backend == "local":
        return OllamaProvider()
    if backend == "bedrock":
        return BedrockProvider(model_id=os.getenv("BEDROCK_MODEL_ID", ""))
    raise ValueError(f"Unknown backend '{backend}'. Expected 'local' or 'bedrock'.")