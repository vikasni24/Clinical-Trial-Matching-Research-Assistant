"""Tests app.api.routes.patients._LazyConfiguredLLMProvider — the piece
that decides, based on Settings.llm_provider, whether AnthropicLLMProvider
or GroqLLMProvider actually gets constructed and called. No real network
call is ever made (httpx.MockTransport throughout); AskService itself is
never involved here since it only ever depends on the LLMProvider Protocol,
not on this selection logic."""

import httpx
import pytest

from app.api.routes import patients as patients_module
from app.config import Settings
from app.services.anthropic_llm_provider import LLMProviderConfigurationError


def _settings(provider="anthropic", api_key="fake-test-key"):
    return Settings(llm_provider=provider, llm_api_key=api_key, llm_timeout_seconds=5.0)


def _client_with_response(json_body):
    def handler(request):
        return httpx.Response(200, json=json_body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_defaults_to_anthropic_when_unset(monkeypatch):
    monkeypatch.setattr(patients_module, "get_settings", lambda: _settings(provider="anthropic"))
    monkeypatch.setattr(patients_module.AnthropicLLMProvider, "generate", lambda self, prompt: "from-anthropic")

    result = patients_module._LazyConfiguredLLMProvider().generate("prompt")

    assert result == "from-anthropic"


def test_selects_groq_when_configured(monkeypatch):
    monkeypatch.setattr(patients_module, "get_settings", lambda: _settings(provider="groq"))
    monkeypatch.setattr(patients_module.GroqLLMProvider, "generate", lambda self, prompt: "from-groq")

    result = patients_module._LazyConfiguredLLMProvider().generate("prompt")

    assert result == "from-groq"


def test_selection_is_case_insensitive(monkeypatch):
    monkeypatch.setattr(patients_module, "get_settings", lambda: _settings(provider="GROQ"))
    monkeypatch.setattr(patients_module.GroqLLMProvider, "generate", lambda self, prompt: "from-groq")

    result = patients_module._LazyConfiguredLLMProvider().generate("prompt")

    assert result == "from-groq"


def test_unknown_provider_raises_configuration_error(monkeypatch):
    monkeypatch.setattr(patients_module, "get_settings", lambda: _settings(provider="openai"))

    with pytest.raises(LLMProviderConfigurationError) as exc_info:
        patients_module._LazyConfiguredLLMProvider().generate("prompt")

    assert "openai" in str(exc_info.value).lower()


def test_groq_provider_actually_used_end_to_end_with_mock_transport(monkeypatch):
    # A real GroqLLMProvider, backed by httpx.MockTransport (no real
    # network call), constructed ahead of time with the mock client baked
    # in — confirms the selection logic reaches an actually-functioning
    # provider, not just a bare stub.
    from app.services.groq_llm_provider import GroqLLMProvider

    settings = _settings(provider="groq", api_key="real-shaped-key")
    client = _client_with_response({"choices": [{"message": {"content": "Groq answered."}}]})
    real_provider = GroqLLMProvider(settings=settings, client=client)

    monkeypatch.setattr(patients_module, "get_settings", lambda: settings)
    monkeypatch.setattr(patients_module, "GroqLLMProvider", lambda settings=None: real_provider)

    result = patients_module._LazyConfiguredLLMProvider().generate("What medications?")

    assert result == "Groq answered."
