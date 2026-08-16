"""Phase 7B: AuditRecord model validation — minimal, deterministic, and
patient-isolated, mirroring the existing validator pattern already used by
RetrievalResult/GroundedContext/GroundedAnswer."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.models.audit import AuditEvidenceReference, AuditRecord, AuditRecordOut


def _reference(patient_id="p1", resource_type="Condition", resource_id="cond-1"):
    return AuditEvidenceReference(patient_id=patient_id, resource_type=resource_type, resource_id=resource_id)


def _record(**overrides):
    defaults = dict(
        audit_id="audit-1",
        patient_id="p1",
        query="hypertension?",
        retrieval_status="evidence_found",
        answer_status="answered",
        evidence_references=[_reference()],
        created_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return AuditRecord(**defaults)


# --- 1: valid construction -------------------------------------------------------------


def test_valid_audit_record_constructs_successfully():
    record = _record()

    assert record.audit_id == "audit-1"
    assert record.patient_id == "p1"
    assert record.retrieval_status == "evidence_found"
    assert record.answer_status == "answered"
    assert record.evidence_references[0].resource_id == "cond-1"


def test_audit_record_with_no_evidence_is_valid():
    record = _record(retrieval_status="no_evidence_found", answer_status="insufficient_evidence", evidence_references=[])

    assert record.evidence_references == []


def test_audit_record_status_values_are_constrained_to_known_states():
    with pytest.raises(ValidationError):
        _record(retrieval_status="made_up_status")

    with pytest.raises(ValidationError):
        _record(answer_status="made_up_status")


# --- 2: empty patient_id rejected -------------------------------------------------------


def test_empty_patient_id_is_rejected():
    with pytest.raises(ValidationError):
        _record(patient_id="")


# --- 3: empty audit_id rejected ----------------------------------------------------------


def test_empty_audit_id_is_rejected():
    with pytest.raises(ValidationError):
        _record(audit_id="")


def test_empty_query_is_rejected():
    with pytest.raises(ValidationError):
        _record(query="")


# --- 4: cross-patient evidence reference rejected -----------------------------------------


def test_cross_patient_evidence_reference_is_rejected():
    with pytest.raises(ValidationError):
        _record(patient_id="patient-a", evidence_references=[_reference(patient_id="patient-b")])


def test_mixed_own_and_cross_patient_evidence_is_rejected():
    with pytest.raises(ValidationError):
        _record(
            patient_id="patient-a",
            evidence_references=[_reference(patient_id="patient-a"), _reference(patient_id="patient-b")],
        )


# --- data minimization: no raw FHIR / secrets / reasoning fields exist on the model -------


def test_audit_record_has_no_raw_fhir_secret_or_reasoning_fields():
    fields = set(AuditRecord.model_fields.keys())
    forbidden = {
        "data",
        "resourceType",
        "raw_answer_text",
        "prompt",
        "api_key",
        "chain_of_thought",
        "reasoning",
        "confidence_score",
        "hallucination_score",
        "medical_probability",
    }
    assert fields.isdisjoint(forbidden)


def test_audit_evidence_reference_carries_only_identifiers_and_patient_scope():
    assert set(AuditEvidenceReference.model_fields.keys()) == {"patient_id", "resource_type", "resource_id"}


# --- API-facing projection strips patient_id from each evidence reference ----------------


def test_audit_record_out_drops_patient_id_from_evidence_references():
    record = _record()

    out = AuditRecordOut.from_record(record)

    assert out.patient_id == "p1"  # kept at the top level
    dumped_reference = out.evidence_references[0].model_dump()
    assert "patient_id" not in dumped_reference
    assert dumped_reference == {"resource_type": "Condition", "resource_id": "cond-1"}
