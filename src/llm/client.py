"""Pluggable LLM client. Provider selected by SCIRAG_LLM_PROVIDER env var."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Common interface for all LLM providers."""

    @abstractmethod
    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_format: str | None = None,
        num_ctx: int | None = None,
    ) -> str:
        """Generate a completion. Returns the assistant's text.

        num_ctx: Ollama-specific context window size. Ignored by hosted
        providers (Anthropic/OpenAI auto-size). Needed for local Llama
        because Ollama's default num_ctx=2048 silently truncates long
        inputs.
        """


class OllamaProvider(LLMClient):
    """Local Ollama provider (default).

    Note: Ollama serializes inference on a single GPU — concurrent calls are
    slower than sequential due to internal queuing. Call generate() sequentially
    in the pipeline (no thread pools).
    """

    def __init__(self, model: str = "llama3.1:8b") -> None:
        self.model = model

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_format: str | None = None,
        num_ctx: int | None = None,
    ) -> str:
        import ollama

        options: dict = {"num_predict": max_tokens, "temperature": temperature}
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": options,
        }
        if response_format == "json":
            kwargs["format"] = "json"

        resp = ollama.chat(**kwargs)
        return resp["message"]["content"]


class AnthropicProvider(LLMClient):
    """Anthropic Claude provider. Install with: pip install scirag[anthropic]

    Reads ANTHROPIC_API_KEY from the environment. Model defaults to Sonnet and
    can be overridden with SCIRAG_ANTHROPIC_MODEL or the `model` argument.
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("SCIRAG_ANTHROPIC_MODEL", "claude-sonnet-4-6")

    def generate(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        response_format: str | None = None,
        num_ctx: int | None = None,
    ) -> str:
        import anthropic

        if response_format == "json":
            system = system + "\n\nRespond with valid JSON only, no prose."
        client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY from env
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class OpenAIProvider(LLMClient):
    """Stub — install with: pip install scirag[openai]"""

    def generate(self, system: str, user: str, **kwargs) -> str:
        raise NotImplementedError(
            "OpenAI provider not yet implemented. Install extras: pip install scirag[openai]"
        )


_PROVIDERS: dict[str, type[LLMClient]] = {
    "ollama": OllamaProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
}


def get_client(provider: str | None = None) -> LLMClient:
    """Return an LLMClient for the requested provider.

    Resolution order: explicit arg > SCIRAG_LLM_PROVIDER env var > 'ollama'.
    """
    name = provider or os.environ.get("SCIRAG_LLM_PROVIDER", "ollama")
    name = name.lower().strip()
    if name not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM provider '{name}'. Choose from: {', '.join(_PROVIDERS)}"
        )
    return _PROVIDERS[name]()
