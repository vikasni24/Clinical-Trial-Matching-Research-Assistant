"""Phase 6A: GroundedAnswer — contract/model only. No LLM, no answer
generation exists yet; these tests exercise the Pydantic model directly."""

import pytest
from pydantic import ValidationError

from app.models.answer import GroundedAnswer
from app.models.evidence import Evidence


def _evidence(patient_id="p1", resource_id="obs-1"):
    return Evidence(
        patient_id=patient_id,
        resource_type="Observation",
        resource_id=resource_id,
        code="8480-6",
        display="Systolic Blood Pressure",
        value=125,
        unit="mm[Hg]",
    )


# --- valid states ------------------------------------------------------------------


def test_valid_answered_state():
    answer = GroundedAnswer(
        patient_id="p1",
        query="What is the patient's blood pressure?",
        status="answered",
        answer_text="The patient's most recent systolic blood pressure was 125 mm[Hg].",
        evidence=[_evidence()],
    )

    assert answer.status == "answered"
    assert answer.answer_text is not None
    assert len(answer.evidence) == 1


def test_valid_insufficient_evidence_state():
    answer = GroundedAnswer(
        patient_id="p1",
        query="What is the patient's HbA1c?",
        status="insufficient_evidence",
        message="No HbA1c evidence was found for this patient.",
    )

    assert answer.status == "insufficient_evidence"
    assert answer.answer_text is None
    assert answer.evidence == []


def test_valid_unsupported_state():
    answer = GroundedAnswer(
        patient_id="p1",
        query="???",
        status="unsupported",
        message="Query contained no meaningful terms to search on.",
    )

    assert answer.status == "unsupported"
    assert answer.answer_text is None
    assert answer.evidence == []


# --- distinctness of the three states -----------------------------------------------


def test_answered_is_distinct_from_insufficient_evidence():
    answered = GroundedAnswer(
        patient_id="p1", query="q", status="answered", answer_text="text", evidence=[_evidence()]
    )
    insufficient = GroundedAnswer(patient_id="p1", query="q", status="insufficient_evidence")

    assert answered.status != insufficient.status
    assert bool(answered.evidence) != bool(insufficient.evidence)


def test_insufficient_evidence_is_distinct_from_unsupported():
    insufficient = GroundedAnswer(patient_id="p1", query="q", status="insufficient_evidence")
    unsupported = GroundedAnswer(patient_id="p1", query="q", status="unsupported")

    assert insufficient.status != unsupported.status


# --- empty patient_id / query rejection ----------------------------------------------


def test_empty_patient_id_rejected():
    with pytest.raises(ValidationError):
        GroundedAnswer(patient_id="", query="q", status="insufficient_evidence")


def test_empty_query_rejected():
    with pytest.raises(ValidationError):
        GroundedAnswer(patient_id="p1", query="", status="insufficient_evidence")


# --- patient/evidence mismatch rejection ----------------------------------------------


def test_patient_evidence_mismatch_rejected():
    other_patient_evidence = _evidence(patient_id="p2")

    with pytest.raises(ValidationError):
        GroundedAnswer(
            patient_id="p1",
            query="q",
            status="answered",
            answer_text="text",
            evidence=[other_patient_evidence],
        )


def test_mixed_patient_evidence_rejected():
    # Even one evidence item from a different patient among otherwise-valid
    # evidence must be rejected — no partial leakage.
    mixed_evidence = [_evidence(patient_id="p1", resource_id="obs-a"), _evidence(patient_id="p2", resource_id="obs-b")]

    with pytest.raises(ValidationError):
        GroundedAnswer(patient_id="p1", query="q", status="answered", answer_text="text", evidence=mixed_evidence)


# --- invalid status/evidence/answer_text combinations ---------------------------------


def test_answered_without_evidence_is_rejected():
    with pytest.raises(ValidationError):
        GroundedAnswer(patient_id="p1", query="q", status="answered", answer_text="text", evidence=[])


def test_answered_without_answer_text_is_rejected():
    with pytest.raises(ValidationError):
        GroundedAnswer(patient_id="p1", query="q", status="answered", answer_text=None, evidence=[_evidence()])


def test_insufficient_evidence_with_evidence_attached_is_rejected():
    with pytest.raises(ValidationError):
        GroundedAnswer(patient_id="p1", query="q", status="insufficient_evidence", evidence=[_evidence()])


def test_unsupported_with_evidence_attached_is_rejected():
    with pytest.raises(ValidationError):
        GroundedAnswer(patient_id="p1", query="q", status="unsupported", evidence=[_evidence()])


def test_insufficient_evidence_with_answer_text_is_rejected():
    # An "insufficient evidence" state must never carry a fabricated answer.
    with pytest.raises(ValidationError):
        GroundedAnswer(patient_id="p1", query="q", status="insufficient_evidence", answer_text="Yes, the patient has it.")


def test_unsupported_with_answer_text_is_rejected():
    with pytest.raises(ValidationError):
        GroundedAnswer(patient_id="p1", query="q", status="unsupported", answer_text="Some fabricated answer.")


# --- prohibited scoring/reasoning fields are absent -----------------------------------


def test_prohibited_fields_are_absent():
    prohibited = {
        "confidence_score",
        "hallucination_score",
        "medical_probability",
        "ai_reasoning",
        "chain_of_thought",
        "reasoning",
        "metadata",
        "probability",
        "certainty",
    }
    assert prohibited.isdisjoint(GroundedAnswer.model_fields.keys())


def test_no_raw_fhir_or_mongo_id_in_serialized_answer():
    answer = GroundedAnswer(
        patient_id="p1", query="q", status="answered", answer_text="text", evidence=[_evidence()]
    )
    dumped = answer.model_dump()

    assert "data" not in dumped
    for evidence_dict in dumped["evidence"]:
        assert "data" not in evidence_dict
        assert "resourceType" not in evidence_dict
        assert "_id" not in evidence_dict
