"""Phase 9 Part G: verifies failures stay isolated at each service boundary
— a failure anywhere in the pipeline must never surface as a fabricated
200 answer, and must never silently corrupt an otherwise-valid response.
No real LLM is ever called."""

import shutil

import pytest
from pymongo.errors import PyMongoError

from app.api.routes.patients import get_llm_provider
from app.main import app
from app.repositories import audit_repository
from app.services.anthropic_llm_provider import LLMProviderRequestError
from app.services.ask_service import AskService
from app.services.fhir_ingestion import FHIRIngestionService
from app.services.hybrid_retriever import HybridEvidenceRetriever
from app.services.patient_normalization import PatientNormalizationService
from app.services.rag_context import GroundedContextService


def _setup(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()
    PatientNormalizationService(mongo_db).normalize_all()


class _FakeLLMProvider:
    def __init__(self, response: str = "unused"):
        self.calls: list[str] = []
        self._response = response

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


def _override_llm(fake):
    app.dependency_overrides[get_llm_provider] = lambda: fake


def _clear_llm_override():
    app.dependency_overrides.pop(get_llm_provider, None)


# --- retrieval failure -> no fabricated answer ---------------------------------------------


def test_retrieval_failure_does_not_produce_a_fabricated_answer(mongo_db, tmp_path, fixtures_dir, monkeypatch):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider()

    def _raise(*args, **kwargs):
        raise PyMongoError("simulated retrieval failure")

    monkeypatch.setattr(HybridEvidenceRetriever, "retrieve", _raise)

    with pytest.raises(PyMongoError):
        AskService(mongo_db, provider).ask("profile-patient-1", "hypertension")

    # The LLM must never be reached when retrieval itself failed, and no
    # audit record describing a nonexistent outcome is written.
    assert provider.calls == []
    _, total = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-1")
    assert total == 0


def test_retrieval_failure_at_the_api_returns_503_not_a_fabricated_200(api_client, mongo_db, tmp_path, fixtures_dir, monkeypatch):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider())

    def _raise(*args, **kwargs):
        raise PyMongoError("simulated database outage during retrieval")

    monkeypatch.setattr(GroundedContextService, "build_context", _raise)

    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})

    _clear_llm_override()
    assert resp.status_code == 503
    assert resp.json() == {"detail": "Database temporarily unavailable"}


# --- safety failure -> prevents generation --------------------------------------------------


def test_safety_gate_failure_prevents_generation(mongo_db, tmp_path, fixtures_dir):
    from app.models.evidence import Evidence
    from app.models.rag_context import GroundedContext
    from app.services.safety_rules import enforce_pre_generation_safety

    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider()

    contaminated_context = GroundedContext.model_construct(
        patient_id="profile-patient-1",
        query="hypertension",
        status="evidence_found",
        evidence=[Evidence(patient_id="profile-patient-2", resource_type="Condition", resource_id="cond-2", display="Diabetes")],
        message=None,
    )

    with pytest.raises(ValueError):
        enforce_pre_generation_safety(contaminated_context)

    # The safety gate raising must mean generation is never attempted by a
    # caller that checks it first (AskService's own real ordering, proven
    # in test_e2e_pipeline.py's scenario 6) — here we additionally confirm
    # the fake provider was never invoked as a standalone object either.
    assert provider.calls == []


# --- LLM failure -> no fake answer -----------------------------------------------------------


def test_llm_failure_does_not_produce_a_fake_answer(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    class _RaisingProvider:
        def generate(self, prompt):
            raise LLMProviderRequestError("simulated provider outage")

    with pytest.raises(LLMProviderRequestError):
        AskService(mongo_db, _RaisingProvider()).ask("profile-patient-1", "hypertension")


def test_llm_failure_at_the_api_returns_502_not_a_fabricated_200(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    class _RaisingProvider:
        def generate(self, prompt):
            raise LLMProviderRequestError("simulated provider outage")

    _override_llm(_RaisingProvider())
    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})
    _clear_llm_override()

    assert resp.status_code == 502
    assert "answer_text" not in resp.json()
    assert "status" not in resp.json() or resp.json().get("status") != "answered"


# --- answer validation failure -> never exposes raw LLM output ------------------------------


def test_answer_validation_rejection_never_exposes_raw_llm_output(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    raw_output = "SECRET-MARKER: patient definitely has [Condition/fabricated-id]."

    answer = AskService(mongo_db, _FakeLLMProvider(raw_output)).ask("profile-patient-1", "hypertension")

    assert answer.status == "insufficient_evidence"
    assert answer.answer_text is None
    assert answer.answer_text != raw_output


# --- audit failure -> does not corrupt an already valid answer ------------------------------


def test_audit_failure_does_not_corrupt_the_returned_answer(api_client, mongo_db, tmp_path, fixtures_dir, monkeypatch):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider("The patient has hypertension [Condition/cond-1]."))

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated audit-write failure")

    monkeypatch.setattr(audit_repository, "insert_audit_record", _raise)

    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})

    _clear_llm_override()
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "answered"
    assert body["evidence"][0]["resource_id"] == "cond-1"


# --- database failure -> no silently fabricated clinical data -------------------------------


def test_database_failure_during_evidence_lookup_does_not_fabricate_data(api_client, mongo_db, tmp_path, fixtures_dir, monkeypatch):
    _setup(mongo_db, tmp_path, fixtures_dir)

    def _raise(*args, **kwargs):
        raise PyMongoError("simulated database outage")

    monkeypatch.setattr(mongo_db["fhir_resources"], "find", _raise)

    resp = api_client.get("/api/patients/profile-patient-1/evidence")

    assert resp.status_code == 503
    assert resp.json() == {"detail": "Database temporarily unavailable"}


# --- patient-not-found -> stops the workflow before retrieval --------------------------------


def test_patient_not_found_stops_before_retrieval_is_ever_attempted(api_client, monkeypatch):
    calls = []
    original = GroundedContextService.build_context

    def _spy(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GroundedContextService, "build_context", _spy)

    resp = api_client.post("/api/patients/does-not-exist/ask", json={"query": "hypertension"})

    assert resp.status_code == 404
    assert calls == []  # retrieval was never reached
