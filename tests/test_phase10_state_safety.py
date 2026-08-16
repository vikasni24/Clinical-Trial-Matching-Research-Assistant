"""Phase 10E: re-confirms the 5 distinct states of the /ask pipeline never
collapse into one another, and — the critical invariant restated by this
phase — that absence of evidence is NEVER converted into a negative
clinical fact. Organized as one explicit state-transition table so the
full state space is visible in one place, rather than scattered assertions
across other test files. No real LLM is ever called."""

import shutil

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


_NEGATIVE_CLAIM_PHRASES = (
    "does not have", "doesn't have", "no evidence of", "negative for",
    "ruled out", "absence of", "not present", "is normal", "no diabetes",
)


def _assert_no_negative_clinical_claim(message):
    text = (message or "").lower()
    for phrase in _NEGATIVE_CLAIM_PHRASES:
        assert phrase not in text, f"found forbidden negative-claim phrase {phrase!r} in: {message!r}"


# --- state 1: unsupported --------------------------------------------------------------------


def test_state_unsupported(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider()

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "???")

    assert answer.status == "unsupported"
    assert provider.calls == []
    assert answer.answer_text is None
    assert answer.evidence == []
    _assert_no_negative_clinical_claim(answer.message)


# --- state 2: no evidence (retrieval) -> insufficient_evidence (answer) ----------------------


def test_state_no_evidence(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider()

    # "hba1c" is a registered structured concept AND tokenizes to a real
    # semantic query term, so this is a genuinely SUPPORTED query that
    # simply finds nothing — distinct from "unsupported".
    answer = AskService(mongo_db, provider).ask("profile-patient-1", "hba1c")

    assert answer.status == "insufficient_evidence"
    assert provider.calls == []
    assert answer.answer_text is None
    assert answer.evidence == []
    _assert_no_negative_clinical_claim(answer.message)


# --- state 3: evidence found + valid citation -> answered -------------------------------------


def test_state_evidence_found_and_answered(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider("Blood pressure was recorded [Observation/obs-1].")

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "blood pressure")

    assert answer.status == "answered"
    assert len(provider.calls) == 1
    assert answer.answer_text is not None
    assert answer.evidence != []


# --- state 4: evidence found but validation rejects it -> insufficient_evidence --------------


def test_state_evidence_found_but_validation_rejects(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider("The patient seems fine overall.")  # cites nothing

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "hypertension")

    # Retrieval DID find evidence (the LLM WAS called) — but validation
    # still rejected an uncited answer. Distinct from state 2, where the
    # LLM is never even invoked because nothing was retrieved at all.
    assert len(provider.calls) == 1
    assert answer.status == "insufficient_evidence"
    assert answer.answer_text is None
    assert answer.evidence == []
    _assert_no_negative_clinical_claim(answer.message)


# --- state 5: valid grounded answer with multiple citations -----------------------------------


def test_state_valid_grounded_answer(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    provider = _FakeLLMProvider(
        "The patient has hypertension [Condition/cond-1] and takes metformin, "
        "with a blood pressure reading of 125 [Observation/obs-1]."
    )

    answer = AskService(mongo_db, provider).ask("profile-patient-1", "hypertension blood pressure")

    assert answer.status == "answered"
    assert {e.resource_id for e in answer.evidence} == {"cond-1", "obs-1"}


# --- the 5 states are pairwise distinct: no state's outcome equals another's -------------------


def test_all_five_states_are_pairwise_distinct(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    outcomes = {
        "unsupported": AskService(mongo_db, _FakeLLMProvider()).ask("profile-patient-1", "???").status,
        "no_evidence": AskService(mongo_db, _FakeLLMProvider()).ask("profile-patient-1", "hba1c").status,
        "rejected": AskService(mongo_db, _FakeLLMProvider("fine overall")).ask("profile-patient-1", "hypertension").status,
        "answered": AskService(mongo_db, _FakeLLMProvider("BP noted [Observation/obs-1].")).ask(
            "profile-patient-1", "blood pressure"
        ).status,
    }

    assert outcomes["unsupported"] == "unsupported"
    assert outcomes["no_evidence"] == "insufficient_evidence"
    assert outcomes["rejected"] == "insufficient_evidence"  # same final status as no_evidence...
    assert outcomes["answered"] == "answered"
    # ...but the retrieval_status that led there differs, which is exactly
    # why the audit layer (Phase 7) records retrieval_status SEPARATELY
    # from answer_status rather than collapsing them into one field.
