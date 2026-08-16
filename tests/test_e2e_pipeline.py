"""Phase 9 Part B + Part D: end-to-end integration tests exercising the
COMPLETE pipeline (patient validation -> HybridEvidenceRetriever ->
GroundedContext -> pre-generation safety -> grounded prompt -> LLMProvider
-> answer validation -> GroundedAnswer -> audit persistence) through
AskService against real ingested FHIR data in mongomock. No real LLM is
ever called — a fake/spy provider stands in throughout.

Distinct from earlier phases' unit tests (test_safety_rules.py,
test_answer_validator.py, test_ask_service_audit.py): those exercise one
layer at a time with hand-built GroundedContext/Evidence objects. Every
test here goes through the real FHIR ingestion -> normalization ->
retrieval chain, so a break at any seam between layers would surface here
even if each layer's own unit tests still pass in isolation.
"""

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
    def __init__(self, response: str = "unused"):
        self.calls: list[str] = []
        self._response = response

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


class _RaisingLLMProvider:
    def __init__(self, exc: Exception):
        self.calls: list[str] = []
        self._exc = exc

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        raise self._exc


# =====================================================================
# Part B — end-to-end pipeline scenarios
# =====================================================================


# --- 1: valid patient + supported query + available evidence -----------------------------


def test_scenario_1_valid_patient_supported_query_available_evidence(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider("The patient has hypertension [Condition/cond-1].")

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "hypertension")

    assert answer.status == "answered"
    assert len(provider.calls) == 1
    # Evidence in the final answer must be traceable back to what was
    # actually retrieved and ingested (cond-1, from the fixture bundle).
    assert [e.resource_id for e in answer.evidence] == ["cond-1"]
    assert answer.evidence[0].patient_id == "profile-patient-1"

    records, total = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-1")
    assert total == 1
    assert records[0].answer_status == "answered"
    assert records[0].retrieval_status == "evidence_found"
    assert [r.resource_id for r in records[0].evidence_references] == ["cond-1"]


# --- 2: valid patient + supported query + no evidence ---------------------------------------


def test_scenario_2_supported_query_no_evidence(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider("The patient's HbA1c is definitely 9.2%.")

    # "hemoglobin a1c" is a registered structured concept, but this patient
    # has no such observation on record.
    answer = AskService(mongo_db, provider).ask("profile-patient-1", "hemoglobin a1c")

    assert answer.status == "insufficient_evidence"
    assert provider.calls == []  # the LLM must never be called
    assert answer.answer_text is None
    assert answer.evidence == []
    # No fabricated clinical conclusion of any kind leaks through.
    assert answer.message is not None
    for forbidden in ("9.2", "does not have", "negative", "ruled out", "normal"):
        assert forbidden not in answer.message.lower()


# --- 3: valid patient + unsupported query -----------------------------------------------------


def test_scenario_3_unsupported_query(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider()

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "???")

    assert answer.status == "unsupported"
    assert provider.calls == []


# --- 4: LLM produces a fabricated citation ----------------------------------------------------


def test_scenario_4_fabricated_citation_is_rejected(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    raw_output = "The patient also has [Condition/does-not-exist-in-mongo], a serious concern."
    provider = _FakeLLMProvider(raw_output)

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "hypertension")

    assert answer.status == "insufficient_evidence"
    assert answer.evidence == []
    assert "does-not-exist-in-mongo" not in [e.resource_id for e in answer.evidence]
    # The raw LLM text is never returned to the caller.
    assert answer.answer_text is None
    assert answer.answer_text != raw_output

    records, _ = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-1")
    assert records[0].answer_status == "insufficient_evidence"
    assert records[0].evidence_references == []


# --- 5: LLM produces valid citations -----------------------------------------------------------


def test_scenario_5_valid_citations_only_context_evidence_is_used(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider("Blood pressure reading: [Observation/obs-1].")

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "blood pressure")

    assert answer.status == "answered"
    assert len(answer.evidence) == 1
    assert answer.evidence[0].resource_id == "obs-1"
    assert answer.evidence[0].resource_type == "Observation"


# --- 6: cross-patient evidence contamination attempt --------------------------------------------


def test_scenario_6_cross_patient_contamination_is_rejected_not_filtered(mongo_db, tmp_path, fixtures_dir):
    from app.models.rag_context import GroundedContext
    from app.services.safety_rules import enforce_pre_generation_safety

    _setup(mongo_db, tmp_path, fixtures_dir)

    # Simulate an invalid upstream condition: GroundedContext's own
    # validator would normally make this impossible to construct, so
    # model_construct() bypasses it (as an upstream bug might).
    from app.models.evidence import Evidence

    contaminated_context = GroundedContext.model_construct(
        patient_id="profile-patient-1",
        query="hypertension",
        status="evidence_found",
        evidence=[
            Evidence(
                patient_id="profile-patient-2",  # a different patient's evidence
                resource_type="Condition",
                resource_id="cond-2",
                display="Diabetes",
            )
        ],
        message=None,
    )

    # The system must REJECT this outright (raise), never silently drop the
    # rogue item and proceed, and never silently accept it into an answer.
    with pytest.raises(ValueError):
        enforce_pre_generation_safety(contaminated_context)


# --- 7: LLM provider failure ---------------------------------------------------------------------


def test_scenario_7_llm_provider_failure_produces_no_fabricated_answer(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _RaisingLLMProvider(LLMProviderRequestError("simulated upstream provider outage"))

    with pytest.raises(LLMProviderRequestError):
        AskService(mongo_db, provider).ask("profile-patient-1", "hypertension")

    # No GroundedAnswer was ever produced -> no misleading audit record.
    _, total = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-1")
    assert total == 0


# --- 8: audit persistence failure -----------------------------------------------------------------


def test_scenario_8_audit_persistence_failure_does_not_change_the_answer(mongo_db, tmp_path, fixtures_dir, monkeypatch):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider("The patient has hypertension [Condition/cond-1].")

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated MongoDB audit-write failure")

    monkeypatch.setattr(audit_repository, "insert_audit_record", _raise)

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "hypertension")

    assert answer.status == "answered"
    assert answer.evidence[0].resource_id == "cond-1"


# =====================================================================
# Part D — UNKNOWN / no-evidence safety
# =====================================================================


def test_no_hba1c_evidence_is_insufficient_not_negative(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    answer = AskService(mongo_db, _FakeLLMProvider()).ask("profile-patient-1", "hba1c")

    assert answer.status == "insufficient_evidence"
    assert "does not have" not in (answer.message or "").lower()


def test_unsupported_concept_is_unsupported_not_negative(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    # "???" tokenizes to nothing meaningful for either retrieval strategy
    # (no registered structured concept, no semantic tokens at all) — the
    # one query shape that is genuinely unsupported by both, per
    # HybridEvidenceRetriever's own truth table (see test_hybrid_retriever.py).
    answer = AskService(mongo_db, _FakeLLMProvider()).ask("profile-patient-1", "???")

    assert answer.status == "unsupported"
    assert "does not have" not in (answer.message or "").lower()


def test_missing_clinical_value_is_never_fabricated_in_the_prompt(mongo_db, tmp_path, fixtures_dir):
    from app.models.evidence import Evidence
    from app.models.rag_context import GroundedContext
    from app.services.grounded_prompt import build_grounded_prompt

    # An Evidence item that carries no value at all (e.g. a condition with
    # no associated measurement) must never have a placeholder value
    # invented for it when rendered into a prompt.
    context = GroundedContext(
        patient_id="p1",
        query="hypertension?",
        status="evidence_found",
        evidence=[Evidence(patient_id="p1", resource_type="Condition", resource_id="cond-1", display="Hypertension")],
    )

    prompt = build_grounded_prompt(context)

    assert "value=" not in prompt.evidence_text
    assert "None" not in prompt.evidence_text


def test_incomplete_evidence_uncited_by_llm_is_insufficient(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    # Evidence IS retrieved (the Condition exists), but the LLM's answer
    # doesn't actually cite anything from it — must not become a confident
    # answer just because SOME evidence existed.
    answer = AskService(mongo_db, _FakeLLMProvider("The patient seems fine overall.")).ask(
        "profile-patient-1", "hypertension"
    )

    assert answer.status == "insufficient_evidence"
    assert answer.evidence == []


def test_patient_with_no_matching_resource_gets_no_evidence_found_not_a_denial(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    # profile-patient-2 has diabetes, not hypertension — a real patient
    # with genuinely no matching resource for this concept.
    answer = AskService(mongo_db, _FakeLLMProvider()).ask("profile-patient-2", "hypertension")

    assert answer.status == "insufficient_evidence"
    assert "does not have" not in (answer.message or "").lower()
    assert "ruled out" not in (answer.message or "").lower()
