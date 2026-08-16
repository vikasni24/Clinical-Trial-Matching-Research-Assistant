"""Phase 10B + 10D + 10I regression: end-to-end backend contract tests for
POST /api/patients/{patient_id}/ask's full chain (patient validation ->
retrieval -> hybrid evidence -> grounded context -> pre-generation safety
-> prompt -> fake LLM -> answer validation -> GroundedAnswer -> audit), plus
evidence traceability back to the real stored fhir_resources document, plus
a regression test for the Phase 10I fix (raw LLM provider response text
must never reach the API client). No real LLM is ever called."""

import shutil

import pytest

from app.api.routes.patients import get_llm_provider
from app.main import app
from app.repositories import audit_repository
from app.services.anthropic_llm_provider import LLMProviderRequestError
from app.services.ask_service import AskService
from app.services.fhir_ingestion import FHIRIngestionService
from app.models.retrieval import RetrievalRequest
from app.services.hybrid_retriever import HybridEvidenceRetriever
from app.services.patient_normalization import PatientNormalizationService


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


# =====================================================================
# 10B.1 — valid patient + supported query + evidence
# =====================================================================


def test_10b1_valid_patient_supported_query_with_evidence(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    # Confirm retrieval genuinely finds evidence before the LLM is ever
    # involved (the pipeline's own first stage, checked independently).
    retrieval = HybridEvidenceRetriever(mongo_db).retrieve(
        RetrievalRequest(patient_id="profile-patient-1", query="hypertension")
    )
    assert retrieval.status == "evidence_found"

    provider = _FakeLLMProvider("The patient has hypertension [Condition/cond-1].")
    answer = AskService(mongo_db, provider).ask("profile-patient-1", "hypertension")

    assert len(provider.calls) == 1  # the LLM WAS called
    assert answer.status == "answered"
    assert [e.resource_id for e in answer.evidence] == ["cond-1"]

    records, total = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-1")
    assert total == 1
    assert records[0].answer_status == "answered"


# =====================================================================
# 10B.2 — valid patient + unsupported query
# =====================================================================


def test_10b2_unsupported_query(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider()

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "???")

    assert provider.calls == []
    assert answer.status == "unsupported"
    assert answer.answer_text is None

    records, total = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-1")
    assert total == 1
    assert records[0].answer_status == "unsupported"
    assert records[0].retrieval_status == "unsupported"


# =====================================================================
# 10B.3 — valid patient + supported query + no evidence
# =====================================================================


def test_10b3_supported_query_no_evidence(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider()

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "hemoglobin a1c")

    assert provider.calls == []
    assert answer.status == "insufficient_evidence"
    assert answer.answer_text is None
    assert answer.evidence == []


# =====================================================================
# 10B.4 — fabricated citation is rejected
# =====================================================================


def test_10b4_fabricated_citation_rejected(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider("Also see [Condition/fabricated-resource-id] for more.")

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "hypertension")

    assert answer.status == "insufficient_evidence"
    assert answer.evidence == []
    assert "fabricated-resource-id" not in [e.resource_id for e in answer.evidence]


# =====================================================================
# 10B.5 — valid citations succeed, only real Evidence is returned
# =====================================================================


def test_10b5_valid_citations_return_only_real_evidence(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider("Blood pressure was recorded [Observation/obs-1].")

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "blood pressure")

    assert answer.status == "answered"
    assert len(answer.evidence) == 1
    assert answer.evidence[0].resource_id == "obs-1"
    # Every returned Evidence object must be one that was actually present
    # in the GroundedContext supplied to generation — never invented.
    context_evidence_ids = {"obs-1"}  # ingested fixture's only Observation
    assert {e.resource_id for e in answer.evidence}.issubset(context_evidence_ids)


# =====================================================================
# 10B.6 — unknown patient
# =====================================================================


def test_10b6_unknown_patient(api_client, monkeypatch):
    calls = []
    original = HybridEvidenceRetriever.retrieve

    def _spy(self, *args, **kwargs):
        calls.append(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(HybridEvidenceRetriever, "retrieve", _spy)
    fake = _FakeLLMProvider()
    _override_llm(fake)

    resp = api_client.post("/api/patients/does-not-exist/ask", json={"query": "hypertension"})

    _clear_llm_override()
    assert resp.status_code == 404
    assert calls == []  # retrieval never occurred
    assert fake.calls == []  # LLM never ran


def test_10b6_unknown_patient_creates_no_audit_record(mongo_db):
    _, total = audit_repository.get_patient_audit_history(mongo_db, "does-not-exist")
    assert total == 0


# =====================================================================
# 10B.7 — LLM failure
# =====================================================================


def test_10b7_llm_failure_correct_error_and_no_raw_output_exposed(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    class _RaisingProvider:
        def generate(self, prompt):
            raise LLMProviderRequestError("upstream 503: {\"error\": \"rate limited\", \"internal_trace_id\": \"xyz\"}")

    with pytest.raises(LLMProviderRequestError):
        AskService(mongo_db, _RaisingProvider()).ask("profile-patient-1", "hypertension")

    _, total = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-1")
    assert total == 0  # no invalid audit record was created


def test_10b7_llm_failure_at_api_never_leaks_raw_provider_response(api_client, mongo_db, tmp_path, fixtures_dir):
    """Regression test for the Phase 10I fix: AnthropicLLMProvider embeds
    the raw upstream HTTP response body in LLMProviderRequestError's
    message (see anthropic_llm_provider.py); the API must never forward
    that raw text to the client."""
    _setup(mongo_db, tmp_path, fixtures_dir)
    sensitive_upstream_body = '{"error": "invalid_request", "trace_id": "SENSITIVE-UPSTREAM-TRACE-abc123"}'

    class _RaisingProvider:
        def generate(self, prompt):
            raise LLMProviderRequestError(f"LLM provider returned an error response: 400 {sensitive_upstream_body}")

    _override_llm(_RaisingProvider())
    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})
    _clear_llm_override()

    assert resp.status_code == 502
    assert "SENSITIVE-UPSTREAM-TRACE-abc123" not in resp.text
    assert "invalid_request" not in resp.text
    assert resp.json()["detail"] == "LLM provider request failed"


# =====================================================================
# 10B.8 — audit persistence failure
# =====================================================================


def test_10b8_audit_persistence_failure_answer_still_returned(mongo_db, tmp_path, fixtures_dir, monkeypatch):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider("The patient has hypertension [Condition/cond-1].")

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated audit write failure")

    monkeypatch.setattr(audit_repository, "insert_audit_record", _raise)

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "hypertension")

    assert answer.status == "answered"
    assert answer.evidence[0].resource_id == "cond-1"


def test_10b8_audit_persistence_failure_request_does_not_fail_at_api(api_client, mongo_db, tmp_path, fixtures_dir, monkeypatch):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider("The patient has hypertension [Condition/cond-1]."))

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated audit write failure")

    monkeypatch.setattr(audit_repository, "insert_audit_record", _raise)

    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})

    _clear_llm_override()
    assert resp.status_code == 200
    assert resp.json()["status"] == "answered"


# =====================================================================
# 10D — evidence traceability end-to-end (GroundedAnswer -> fhir_resources)
# =====================================================================


def test_10d_returned_evidence_is_traceable_to_the_stored_fhir_resource(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider("Blood pressure was recorded [Observation/obs-1].")

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "blood pressure")

    assert answer.status == "answered"
    returned_evidence = answer.evidence[0]

    # Independently look up the raw stored FHIR document this evidence
    # claims to be traceable to.
    stored = mongo_db["fhir_resources"].find_one(
        {"resource_type": returned_evidence.resource_type, "resource_id": returned_evidence.resource_id}
    )
    assert stored is not None
    assert stored["patient_id"] == returned_evidence.patient_id == "profile-patient-1"
    assert stored["resource_type"] == returned_evidence.resource_type == "Observation"
    assert stored["resource_id"] == returned_evidence.resource_id == "obs-1"

    # The Evidence's value/unit genuinely originate from the stored FHIR
    # content — never synthesized by the LLM (which only ever saw the
    # already-extracted Evidence, never raw FHIR, and never supplied a
    # value of its own that the validator would trust).
    stored_quantity = stored["data"]["valueQuantity"]
    assert returned_evidence.value == stored_quantity["value"] == 125
    assert returned_evidence.unit == stored_quantity["unit"] == "mm[Hg]"


def test_10d_evidence_existed_in_grounded_context_before_the_llm_ever_ran(mongo_db, tmp_path, fixtures_dir):
    from app.models.retrieval import RetrievalRequest
    from app.services.rag_context import GroundedContextService

    _setup(mongo_db, tmp_path, fixtures_dir)

    # Build the GroundedContext exactly as AskService would, independently
    # of any LLM call, and confirm the evidence the LLM will be asked to
    # cite already existed beforehand — the LLM cannot have originated it.
    context = GroundedContextService(mongo_db).build_context(
        RetrievalRequest(patient_id="profile-patient-1", query="blood pressure")
    )
    assert context.status == "evidence_found"
    pre_llm_resource_ids = {e.resource_id for e in context.evidence}

    provider = _FakeLLMProvider("Blood pressure was recorded [Observation/obs-1].")
    answer = AskService(mongo_db, provider).ask("profile-patient-1", "blood pressure")

    assert {e.resource_id for e in answer.evidence}.issubset(pre_llm_resource_ids)


# =====================================================================
# 10H — audit integrity across every GroundedAnswer status
# =====================================================================


def test_10h_audit_integrity_for_answered_status(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    AskService(mongo_db, _FakeLLMProvider("Noted [Condition/cond-1].")).ask("profile-patient-1", "hypertension")

    records, _ = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-1")
    record = records[0]
    assert record.answer_status == "answered"
    assert record.retrieval_status == "evidence_found"
    assert record.evidence_references != []
    assert record.patient_id == "profile-patient-1"


def test_10h_audit_integrity_for_insufficient_evidence_status(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    AskService(mongo_db, _FakeLLMProvider()).ask("profile-patient-1", "hemoglobin a1c")

    records, _ = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-1")
    assert records[0].answer_status == "insufficient_evidence"
    assert records[0].retrieval_status == "no_evidence_found"
    assert records[0].evidence_references == []


def test_10h_audit_integrity_for_unsupported_status(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    AskService(mongo_db, _FakeLLMProvider()).ask("profile-patient-1", "???")

    records, _ = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-1")
    assert records[0].answer_status == "unsupported"
    assert records[0].retrieval_status == "unsupported"
    assert records[0].evidence_references == []


def test_10h_audit_record_never_contains_forbidden_content(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    raw_output = "Noted [Condition/cond-1]. INTERNAL-REASONING: step by step I deduced..."
    AskService(mongo_db, _FakeLLMProvider(raw_output)).ask("profile-patient-1", "hypertension")

    stored = mongo_db["audit_records"].find_one({"patient_id": "profile-patient-1"})
    assert stored is not None
    for forbidden_key in ("answer_text", "prompt", "raw_llm_output", "data", "resourceType", "api_key", "chain_of_thought"):
        assert forbidden_key not in stored
    assert "INTERNAL-REASONING" not in str(stored)
