"""Phase 7E: AskService's audit integration — the audit write must happen
strictly after answer validation, must never change the returned
GroundedAnswer, and must never let a secondary (audit) failure affect the
primary response path. No real LLM is ever called."""

import shutil

import pytest

from app.repositories import audit_repository
from app.services.anthropic_llm_provider import LLMProviderRequestError
from app.services.ask_service import AskService
from app.services.fhir_ingestion import FHIRIngestionService
from app.services.patient_normalization import PatientNormalizationService


def _setup(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()
    PatientNormalizationService(mongo_db).normalize_all()


class _FakeLLMProvider:
    def __init__(self, response: str = "The patient appears generally healthy."):
        self.calls: list[str] = []
        self._response = response

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


class _RaisingLLMProvider:
    def __init__(self, exc: Exception):
        self._exc = exc

    def generate(self, prompt: str) -> str:
        raise self._exc


# --- 12: audit created after successful answer validation --------------------------------


def test_ask_service_creates_audit_after_successful_answer(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider("The patient has hypertension [Condition/cond-1].")

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "hypertension")

    assert answer.status == "answered"
    records, total = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-1")
    assert total == 1
    assert records[0].answer_status == "answered"
    assert records[0].retrieval_status == "evidence_found"
    assert [r.resource_id for r in records[0].evidence_references] == ["cond-1"]
    assert records[0].query == "hypertension"


# --- 13: unsupported query creates the correct audit state --------------------------------


def test_unsupported_query_creates_correct_audit_state(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider()

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "???")

    assert answer.status == "unsupported"
    assert provider.calls == []  # no LLM call for an unsupported query
    records, total = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-1")
    assert total == 1
    assert records[0].retrieval_status == "unsupported"
    assert records[0].answer_status == "unsupported"
    assert records[0].evidence_references == []


# --- 14: no-evidence query creates the correct audit state --------------------------------


def test_no_evidence_query_creates_correct_audit_state(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider()

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "hemoglobin a1c")

    assert answer.status == "insufficient_evidence"
    assert provider.calls == []  # no LLM call when nothing was retrieved
    records, total = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-1")
    assert total == 1
    assert records[0].retrieval_status == "no_evidence_found"
    assert records[0].answer_status == "insufficient_evidence"
    assert records[0].evidence_references == []


# --- 15: fabricated citation cannot become a successful audit -----------------------------


def test_fabricated_citation_cannot_become_a_successful_audit(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider("See [Condition/does-not-exist] for details.")

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "hypertension")

    assert answer.status == "insufficient_evidence"
    records, total = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-1")
    assert total == 1
    # Retrieval DID find evidence; the answer just wasn't validly grounded
    # in it — the audit correctly distinguishes the two.
    assert records[0].retrieval_status == "evidence_found"
    assert records[0].answer_status == "insufficient_evidence"
    assert records[0].evidence_references == []


# --- 16: LLM failure does not expose raw output (and writes no audit) ---------------------


def test_llm_failure_does_not_expose_raw_output_or_create_an_audit(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _RaisingLLMProvider(LLMProviderRequestError("simulated provider outage"))

    with pytest.raises(LLMProviderRequestError):
        AskService(mongo_db, provider).ask("profile-patient-1", "hypertension")

    # No GroundedAnswer was ever produced, so no audit record describing a
    # (nonexistent) outcome is written either.
    _, total = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-1")
    assert total == 0


# --- 17: audit persistence failure does not bypass answer safety --------------------------


def test_audit_persistence_failure_does_not_change_the_returned_answer(mongo_db, tmp_path, fixtures_dir, monkeypatch):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider("The patient has hypertension [Condition/cond-1].")

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated MongoDB write failure")

    monkeypatch.setattr(audit_repository, "insert_audit_record", _raise)

    # The exact same, already-validated answer is still returned — the
    # caller sees no difference whether or not the audit write succeeded.
    answer = AskService(mongo_db, provider).ask("profile-patient-1", "hypertension")

    assert answer.status == "answered"
    assert answer.answer_text == "The patient has hypertension [Condition/cond-1]."
    assert answer.evidence[0].resource_id == "cond-1"


def test_audit_persistence_failure_does_not_raise_out_of_ask_service(mongo_db, tmp_path, fixtures_dir, monkeypatch):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider("The patient has hypertension [Condition/cond-1].")

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated MongoDB write failure")

    monkeypatch.setattr(audit_repository, "insert_audit_record", _raise)

    # No exception should propagate out of ask() even though the audit
    # write underneath it failed.
    AskService(mongo_db, provider).ask("profile-patient-1", "hypertension")


def test_audit_persistence_failure_is_isolated_for_short_circuited_answers_too(
    mongo_db, tmp_path, fixtures_dir, monkeypatch
):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider()

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated MongoDB write failure")

    monkeypatch.setattr(audit_repository, "insert_audit_record", _raise)

    # Unsupported query short-circuits before any LLM call; audit failure
    # here must be equally harmless.
    answer = AskService(mongo_db, provider).ask("profile-patient-1", "???")

    assert answer.status == "unsupported"
    assert provider.calls == []
