"""Tests for src/llm/client.py — provider routing and stubs."""

import os

import pytest

from src.llm.client import (
    AnthropicProvider,
    OllamaProvider,
    OpenAIProvider,
    get_client,
)


# --- get_client routing ---


def test_default_provider_is_ollama(monkeypatch):
    monkeypatch.delenv("SCIRAG_LLM_PROVIDER", raising=False)
    client = get_client()
    assert isinstance(client, OllamaProvider)


def test_env_var_selects_provider(monkeypatch):
    monkeypatch.setenv("SCIRAG_LLM_PROVIDER", "ollama")
    assert isinstance(get_client(), OllamaProvider)

    monkeypatch.setenv("SCIRAG_LLM_PROVIDER", "anthropic")
    assert isinstance(get_client(), AnthropicProvider)

    monkeypatch.setenv("SCIRAG_LLM_PROVIDER", "openai")
    assert isinstance(get_client(), OpenAIProvider)


def test_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("SCIRAG_LLM_PROVIDER", "openai")
    client = get_client(provider="ollama")
    assert isinstance(client, OllamaProvider)


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider 'bogus'"):
        get_client(provider="bogus")


def test_provider_name_case_insensitive():
    client = get_client(provider="OLLAMA")
    assert isinstance(client, OllamaProvider)


# --- Stub providers raise NotImplementedError ---


def test_anthropic_stub_raises():
    with pytest.raises(NotImplementedError, match="Anthropic provider not yet implemented"):
        AnthropicProvider().generate(system="test", user="test")


def test_openai_stub_raises():
    with pytest.raises(NotImplementedError, match="OpenAI provider not yet implemented"):
        OpenAIProvider().generate(system="test", user="test")


# --- OllamaProvider defaults ---


def test_ollama_provider_default_model():
    provider = OllamaProvider()
    assert provider.model == "llama3.1:8b"


def test_ollama_provider_custom_model():
    provider = OllamaProvider(model="mistral:7b")
    assert provider.model == "mistral:7b"


# --- Integration (requires running Ollama) ---


@pytest.mark.slow
def test_ollama_generate_integration():
    """Smoke test — requires Ollama running with llama3.1:8b loaded."""
    client = get_client(provider="ollama")
    result = client.generate(
        system="Reply in exactly one word.",
        user="What color is the sky?",
        max_tokens=16,
        temperature=0.0,
    )
    assert isinstance(result, str)
    assert len(result) > 0
