"""Phase 6C: LLMProvider — a provider-independent interface only. No real
provider exists anywhere in the project. The deterministic fake below is a
test-only utility (per the phase spec, it does NOT live in app/), used
solely to prove the interface shape is usable and that Phase 6B's
GroundedPrompt content can be passed through it unchanged."""

import inspect

import pytest

from app.models.evidence import Evidence
from app.models.rag_context import GroundedContext
from app.services.grounded_prompt import build_grounded_prompt
from app.services.llm_provider import LLMProvider


class _FakeLLMProvider:
    """A deterministic, offline, test-only stand-in for a real LLMProvider.
    Never calls a network or vendor SDK — it deterministically echoes a
    transformation of the input so tests can assert on exact output."""

    def generate(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("prompt must not be empty")
        return f"[FAKE-RESPONSE] {prompt}"


# --- interface shape ----------------------------------------------------------------


def test_llm_provider_protocol_defines_generate_method():
    assert hasattr(LLMProvider, "generate")


def test_llm_provider_protocol_has_no_other_required_members():
    # Keep the interface minimal — exactly what the phase asked for.
    public_members = [name for name in vars(LLMProvider) if not name.startswith("_")]
    assert public_members == ["generate"]


# --- deterministic fake provider compatibility ---------------------------------------


def test_fake_provider_satisfies_the_interface_shape():
    provider: LLMProvider = _FakeLLMProvider()

    result = provider.generate("Hello")

    assert isinstance(result, str)
    assert result == "[FAKE-RESPONSE] Hello"


def test_fake_provider_is_deterministic():
    provider = _FakeLLMProvider()

    first = provider.generate("What is the patient's blood pressure?")
    second = provider.generate("What is the patient's blood pressure?")

    assert first == second


def test_fake_provider_different_prompts_are_independent():
    provider = _FakeLLMProvider()

    a = provider.generate("prompt A")
    b = provider.generate("prompt B")

    assert a != b


# --- empty prompt handling ------------------------------------------------------------


def test_empty_prompt_is_rejected_by_the_fake_provider():
    provider = _FakeLLMProvider()

    with pytest.raises(ValueError):
        provider.generate("")


# --- no provider-specific dependency / no network calls -------------------------------


def test_llm_provider_module_has_no_vendor_or_network_dependency():
    import app.services.llm_provider as module

    imports = [
        line.strip()
        for line in inspect.getsource(module).splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    import_text = "\n".join(imports).lower()
    for forbidden in (
        "openai", "anthropic", "google.generativeai", "genai", "requests",
        "httpx", "urllib", "socket", "aiohttp", "langchain",
    ):
        assert forbidden not in import_text

    # Only the stdlib `typing` module is imported.
    for line in imports:
        assert line.startswith(("from typing", "from __future__"))


def test_llm_provider_module_defines_no_api_key_or_credential_handling():
    # The module docstring legitimately says "requires NO API key" as a
    # disclaimer — so this checks actual code (assignments/parameters),
    # not docstring prose, for real credential-handling patterns.
    import app.services.llm_provider as module

    code_lines = [
        line
        for line in inspect.getsource(module).splitlines()
        if not line.strip().startswith(("#", '"""', "'''")) and '"""' not in line
    ]
    code_text = "\n".join(code_lines).lower()
    for forbidden in ("api_key", "secret", "token=", "bearer"):
        assert forbidden not in code_text


# --- Phase 6B compatibility: GroundedPrompt content passes through unchanged ----------


def test_grounded_prompt_content_can_be_passed_to_the_provider_unchanged():
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

    # Assembling GroundedPrompt's existing string fields into a single
    # prompt string requires no modification to GroundedPrompt itself.
    prompt_text = "\n\n".join([prompt.instructions, prompt.status_note, prompt.evidence_text])

    provider: LLMProvider = _FakeLLMProvider()
    result = provider.generate(prompt_text)

    # The provider received the prompt content verbatim — nothing was
    # altered, truncated, or reinterpreted before being passed through.
    assert prompt.instructions in result
    assert prompt.status_note in result
    assert prompt.evidence_text in result
    assert "[Observation/obs-1]" in result


def test_grounded_prompt_no_evidence_case_can_also_be_passed_through():
    context = GroundedContext(patient_id="p1", query="hemoglobin a1c?", status="no_evidence_found")
    prompt = build_grounded_prompt(context)
    prompt_text = "\n\n".join([prompt.instructions, prompt.status_note, prompt.evidence_text])

    provider = _FakeLLMProvider()
    result = provider.generate(prompt_text)

    assert "insufficient" in result.lower()
