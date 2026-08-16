"""Phase 10G + 10J: the LLM provider abstraction boundary and
configuration/environment handling. Confirms application code depends only
on the LLMProvider Protocol (never a vendor SDK directly, except inside the
one concrete implementation), that provider errors are handled distinctly,
that tests can always inject a fake provider, and that secrets are
configuration-driven, never hardcoded. No real LLM is ever called."""

import inspect
import shutil

import httpx
import pytest

from app.api.routes.patients import get_llm_provider
from app.config import Settings
from app.main import app
from app.services.anthropic_llm_provider import (
    AnthropicLLMProvider,
    LLMProviderConfigurationError,
    LLMProviderRequestError,
)
from app.services.ask_service import AskService
from app.services.fhir_ingestion import FHIRIngestionService
from app.services.llm_provider import LLMProvider
from app.services.patient_normalization import PatientNormalizationService


def _setup(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()
    PatientNormalizationService(mongo_db).normalize_all()


def _override_llm(fake):
    app.dependency_overrides[get_llm_provider] = lambda: fake


def _clear_llm_override():
    app.dependency_overrides.pop(get_llm_provider, None)


# --- 10G: no application code depends on a vendor SDK except the provider itself -------------


def test_ask_service_imports_only_the_provider_protocol_not_a_vendor_sdk():
    import app.services.ask_service as module

    source = inspect.getsource(module)
    assert "import httpx" not in source
    assert "anthropic_llm_provider" not in source
    assert "from app.services.llm_provider import LLMProvider" in source


def test_only_anthropic_llm_provider_imports_httpx_for_llm_calls():
    # httpx itself is a general-purpose dependency (also used by the
    # FastAPI test client) — the check is that no OTHER service module
    # imports it specifically to talk to an LLM vendor.
    import app.services.ask_service as ask_module
    import app.services.grounded_prompt as prompt_module
    import app.services.safety_rules as safety_module

    for module in (ask_module, prompt_module, safety_module):
        source = inspect.getsource(module)
        assert "httpx" not in source


# --- 10G: AskService depends on the Protocol, satisfied by any duck-typed object -------------


def test_ask_service_accepts_any_object_satisfying_the_llm_provider_protocol(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    class _ArbitraryDuckTypedProvider:
        """No inheritance from anything — purely structural typing."""

        def generate(self, prompt: str) -> str:
            return "The patient has hypertension [Condition/cond-1]."

    provider: LLMProvider = _ArbitraryDuckTypedProvider()
    answer = AskService(mongo_db, provider).ask("profile-patient-1", "hypertension")

    assert answer.status == "answered"


# --- 10G: provider configuration errors are handled distinctly from request errors -----------


def test_provider_configuration_error_returns_500_not_502(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    class _MisconfiguredProvider:
        def generate(self, prompt: str) -> str:
            raise LLMProviderConfigurationError("LLM_API_KEY is not configured")

    _override_llm(_MisconfiguredProvider())
    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})
    _clear_llm_override()

    assert resp.status_code == 500


def test_provider_request_error_returns_502_not_500(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    class _FailingProvider:
        def generate(self, prompt: str) -> str:
            raise LLMProviderRequestError("simulated network failure")

    _override_llm(_FailingProvider())
    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})
    _clear_llm_override()

    assert resp.status_code == 502


# --- 10G: raw provider responses never bypass AnswerValidator ---------------------------------


def test_raw_provider_response_always_passes_through_answer_validator(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    class _UntrustworthyProvider:
        def generate(self, prompt: str) -> str:
            # Cites nothing real — if this ever reached the client
            # unvalidated, it would be a fabricated, confident-sounding answer.
            return "The patient definitely has stage 4 cancer with 95% confidence."

    answer = AskService(mongo_db, _UntrustworthyProvider()).ask("profile-patient-1", "hypertension")

    assert answer.status == "insufficient_evidence"
    assert answer.answer_text is None
    assert answer.evidence == []


# --- 10J: no hardcoded credentials ------------------------------------------------------------


def test_no_hardcoded_api_key_anywhere_in_provider_module():
    import app.services.anthropic_llm_provider as module

    code_lines = [
        line
        for line in inspect.getsource(module).splitlines()
        if not line.strip().startswith(("#", '"""', "'''")) and '"""' not in line
    ]
    code_text = "\n".join(code_lines)
    assert "sk-ant-" not in code_text
    assert "sk-" not in code_text


def test_llm_api_key_has_no_default_value():
    field = Settings.model_fields["llm_api_key"]
    assert field.default is None


def test_settings_are_environment_driven_not_hardcoded():
    settings_a = Settings(llm_api_key="key-a")
    settings_b = Settings(llm_api_key="key-b")

    assert settings_a.llm_api_key == "key-a"
    assert settings_b.llm_api_key == "key-b"
    provider_a = AnthropicLLMProvider(settings=settings_a, client=httpx.Client())
    provider_b = AnthropicLLMProvider(settings=settings_b, client=httpx.Client())
    assert provider_a._api_key != provider_b._api_key


# --- 10J: dev/test configuration runs without any real LLM ------------------------------------


def test_full_suite_runs_without_a_real_llm_api_key(mongo_db, tmp_path, fixtures_dir):
    """Sanity check for the whole test environment's design: constructing
    and using AskService requires no real Settings/API key at all when a
    fake provider is injected — proving dev/test workflows never need real
    LLM credentials."""
    _setup(mongo_db, tmp_path, fixtures_dir)

    class _NoCredentialsNeededProvider:
        def generate(self, prompt: str) -> str:
            return "Noted [Condition/cond-1]."

    answer = AskService(mongo_db, _NoCredentialsNeededProvider()).ask("profile-patient-1", "hypertension")
    assert answer.status == "answered"


def test_missing_api_key_fails_fast_with_a_safe_message():
    with pytest.raises(LLMProviderConfigurationError) as exc_info:
        AnthropicLLMProvider(settings=Settings(llm_api_key=None))

    # The error explains that configuration is missing without ever
    # embedding a (nonexistent) key value.
    assert "LLM_API_KEY" in str(exc_info.value)
