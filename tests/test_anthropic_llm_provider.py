"""Phase 6D: AnthropicLLMProvider — the one concrete LLMProvider
implementation. Every test here uses httpx.MockTransport, which intercepts
requests at the transport layer — no real network call is ever made."""

import inspect

import httpx
import pytest

from app.config import Settings
from app.models.evidence import Evidence
from app.models.rag_context import GroundedContext
from app.services.anthropic_llm_provider import (
    AnthropicLLMProvider,
    LLMProviderConfigurationError,
    LLMProviderRequestError,
)
from app.services.grounded_prompt import build_grounded_prompt
from app.services.llm_provider import LLMProvider


def _settings(api_key="fake-test-key", model="test-model", timeout=5.0):
    return Settings(llm_api_key=api_key, llm_model=model, llm_timeout_seconds=timeout)


def _client_with_response(status_code=200, json_body=None, text=None):
    def handler(request):
        if text is not None:
            return httpx.Response(status_code, text=text)
        return httpx.Response(status_code, json=json_body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def _client_that_raises(exc):
    def handler(request):
        raise exc

    return httpx.Client(transport=httpx.MockTransport(handler))


# --- interface conformance ---------------------------------------------------------


def test_satisfies_llm_provider_protocol():
    client = _client_with_response(json_body={"content": [{"text": "Answer"}]})
    provider: LLMProvider = AnthropicLLMProvider(settings=_settings(), client=client)

    result = provider.generate("some prompt")

    assert isinstance(result, str)
    assert result == "Answer"


# --- missing API key -----------------------------------------------------------------


def test_missing_api_key_raises_configuration_error_at_construction():
    with pytest.raises(LLMProviderConfigurationError):
        AnthropicLLMProvider(settings=_settings(api_key=None))


def test_missing_api_key_fails_fast_before_any_request():
    # Configuration errors happen at __init__, never deferred to generate().
    captured_requests = []

    def handler(request):
        captured_requests.append(request)
        return httpx.Response(200, json={"content": [{"text": "x"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))

    with pytest.raises(LLMProviderConfigurationError):
        AnthropicLLMProvider(settings=_settings(api_key=None), client=client)

    assert captured_requests == []


# --- empty API key string is also treated as missing ----------------------------------


def test_empty_string_api_key_is_also_rejected():
    with pytest.raises(LLMProviderConfigurationError):
        AnthropicLLMProvider(settings=_settings(api_key=""))


# --- network failure -------------------------------------------------------------------


def test_network_failure_raises_request_error():
    client = _client_that_raises(httpx.ConnectError("connection refused"))
    provider = AnthropicLLMProvider(settings=_settings(), client=client)

    with pytest.raises(LLMProviderRequestError):
        provider.generate("some prompt")


# --- timeout -----------------------------------------------------------------------------


def test_timeout_raises_request_error():
    client = _client_that_raises(httpx.TimeoutException("timed out"))
    provider = AnthropicLLMProvider(settings=_settings(), client=client)

    with pytest.raises(LLMProviderRequestError):
        provider.generate("some prompt")


# --- provider error (non-200 response) --------------------------------------------------


def test_provider_error_response_raises_request_error():
    client = _client_with_response(status_code=500, text="internal server error")
    provider = AnthropicLLMProvider(settings=_settings(), client=client)

    with pytest.raises(LLMProviderRequestError):
        provider.generate("some prompt")


def test_provider_auth_error_raises_request_error():
    client = _client_with_response(status_code=401, text="invalid API key")
    provider = AnthropicLLMProvider(settings=_settings(), client=client)

    with pytest.raises(LLMProviderRequestError):
        provider.generate("some prompt")


def test_unexpected_response_shape_raises_request_error():
    client = _client_with_response(status_code=200, json_body={"unexpected": "shape"})
    provider = AnthropicLLMProvider(settings=_settings(), client=client)

    with pytest.raises(LLMProviderRequestError):
        provider.generate("some prompt")


# --- returns only generated text, nothing else -------------------------------------------


def test_returns_only_generated_text():
    client = _client_with_response(json_body={"content": [{"text": "The patient's BP is 125 mm[Hg]."}]})
    provider = AnthropicLLMProvider(settings=_settings(), client=client)

    result = provider.generate("What is the patient's blood pressure?")

    assert result == "The patient's BP is 125 mm[Hg]."
    assert isinstance(result, str)


def test_empty_prompt_is_rejected_without_a_request():
    captured_requests = []

    def handler(request):
        captured_requests.append(request)
        return httpx.Response(200, json={"content": [{"text": "x"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = AnthropicLLMProvider(settings=_settings(), client=client)

    with pytest.raises(ValueError):
        provider.generate("")

    assert captured_requests == []


# --- credentials never hardcoded / configured only via Settings ---------------------------


def test_api_key_comes_from_settings_not_hardcoded():
    client = _client_with_response(json_body={"content": [{"text": "ok"}]})
    provider_a = AnthropicLLMProvider(settings=_settings(api_key="key-a"), client=client)
    provider_b = AnthropicLLMProvider(settings=_settings(api_key="key-b"), client=client)

    assert provider_a._api_key == "key-a"
    assert provider_b._api_key == "key-b"


def test_module_source_contains_no_hardcoded_credential():
    import app.services.anthropic_llm_provider as module

    code_lines = [
        line
        for line in inspect.getsource(module).splitlines()
        if not line.strip().startswith(("#", '"""', "'''")) and '"""' not in line
    ]
    code_text = "\n".join(code_lines)
    # No literal string that looks like a real API key was ever embedded.
    assert "sk-ant-" not in code_text
    assert "sk-" not in code_text


# --- never performs its own retrieval / no MongoDB access ---------------------------------


def test_provider_module_imports_no_database_or_retrieval_code():
    import app.services.anthropic_llm_provider as module

    imports = [
        line.strip()
        for line in inspect.getsource(module).splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    import_text = "\n".join(imports).lower()
    for forbidden in (
        "pymongo", "app.db.mongodb", "app.repositories", "evidence_service",
        "evidence_repository", "hybrid_retriever", "structured_retriever", "semantic_retriever",
    ):
        assert forbidden not in import_text


# --- receives an already-built grounded prompt unchanged ----------------------------------


def test_receives_already_built_grounded_prompt_verbatim():
    context = GroundedContext(
        patient_id="p1",
        query="What is the patient's blood pressure?",
        status="evidence_found",
        evidence=[
            Evidence(
                patient_id="p1",
                resource_type="Observation",
                resource_id="obs-1",
                code="8480-6",
                display="Systolic Blood Pressure",
                value=125,
                unit="mm[Hg]",
            )
        ],
    )
    prompt = build_grounded_prompt(context)
    prompt_text = "\n\n".join([prompt.instructions, prompt.status_note, prompt.evidence_text])

    received_payloads = []

    def handler(request):
        import json

        received_payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"content": [{"text": "answer"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = AnthropicLLMProvider(settings=_settings(), client=client)

    provider.generate(prompt_text)

    sent_content = received_payloads[0]["messages"][0]["content"]
    assert sent_content == prompt_text
    assert "[Observation/obs-1]" in sent_content


# --- deterministic mocked behavior, no real network call -----------------------------------


def test_no_real_network_call_is_ever_made(monkeypatch):
    # Guard: fail loudly if anything in this module ever tries to open a
    # real socket during tests, regardless of the mocked transport above.
    import socket

    def _blocked(*args, **kwargs):
        raise AssertionError("A real network call was attempted during tests")

    monkeypatch.setattr(socket, "socket", _blocked)

    client = _client_with_response(json_body={"content": [{"text": "ok"}]})
    provider = AnthropicLLMProvider(settings=_settings(), client=client)

    result = provider.generate("prompt")

    assert result == "ok"
