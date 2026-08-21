"""GroundedAnswer.retrieval_status — an optional, additive field mirroring
AuditRecord's existing retrieval_status/answer_status split. Lets a caller
(the frontend) distinguish "no evidence was ever retrieved" from "evidence
existed but the LLM's raw answer wasn't grounded in it" — both of which
otherwise collapse into the same status="insufficient_evidence". Optional
and defaulted to None specifically so no other existing GroundedAnswer
construction site anywhere in the test suite needed to change."""

from app.models.answer import GroundedAnswer
from app.models.evidence import Evidence
from app.models.rag_context import GroundedContext
from app.services.answer_validator import build_grounded_answer
from app.services.safety_rules import enforce_pre_generation_safety


def _evidence(resource_id="obs-1", patient_id="p1"):
    return Evidence(patient_id=patient_id, resource_type="Observation", resource_id=resource_id, display="Blood Pressure")


# --- field itself is optional and additive --------------------------------------------


def test_retrieval_status_defaults_to_none_when_not_supplied():
    answer = GroundedAnswer(patient_id="p1", query="q", status="insufficient_evidence")
    assert answer.retrieval_status is None


def test_retrieval_status_can_be_set_explicitly():
    answer = GroundedAnswer(patient_id="p1", query="q", status="insufficient_evidence", retrieval_status="no_evidence_found")
    assert answer.retrieval_status == "no_evidence_found"


# --- safety_rules.py populates it correctly ---------------------------------------------


def test_safety_gate_no_evidence_sets_retrieval_status():
    context = GroundedContext(patient_id="p1", query="hba1c?", status="no_evidence_found")
    answer = enforce_pre_generation_safety(context)
    assert answer.retrieval_status == "no_evidence_found"
    assert answer.status == "insufficient_evidence"


def test_safety_gate_unsupported_sets_retrieval_status():
    context = GroundedContext(patient_id="p1", query="???", status="unsupported")
    answer = enforce_pre_generation_safety(context)
    assert answer.retrieval_status == "unsupported"
    assert answer.status == "unsupported"


# --- answer_validator.py populates it correctly, distinguishing the two ------------------
# ways an answer can end up "insufficient_evidence" ---------------------------------------


def test_uncited_answer_has_evidence_found_retrieval_status_despite_insufficient_answer():
    # THE key distinction this field exists for: retrieval genuinely found
    # evidence (retrieval_status="evidence_found"), but the LLM's answer
    # wasn't grounded in it (status="insufficient_evidence") — a
    # fundamentally different situation from no evidence ever existing.
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[_evidence()])

    answer = build_grounded_answer(context, "The patient appears generally healthy.")

    assert answer.status == "insufficient_evidence"
    assert answer.retrieval_status == "evidence_found"


def test_fabricated_citation_has_evidence_found_retrieval_status():
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[_evidence()])

    answer = build_grounded_answer(context, "See [Observation/does-not-exist].")

    assert answer.status == "insufficient_evidence"
    assert answer.retrieval_status == "evidence_found"


def test_answered_state_has_evidence_found_retrieval_status():
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[_evidence()])

    answer = build_grounded_answer(context, "Noted [Observation/obs-1].")

    assert answer.status == "answered"
    assert answer.retrieval_status == "evidence_found"


def test_genuinely_no_evidence_has_no_evidence_found_retrieval_status():
    context = GroundedContext(patient_id="p1", query="hba1c?", status="no_evidence_found")

    answer = build_grounded_answer(context, "irrelevant — never trusted")

    assert answer.status == "insufficient_evidence"
    assert answer.retrieval_status == "no_evidence_found"


def test_no_evidence_and_uncited_answer_are_distinguishable_via_retrieval_status():
    # The exact scenario that motivated this field: two different
    # AskService runs both end up status="insufficient_evidence", but
    # retrieval_status tells them apart.
    no_evidence_context = GroundedContext(patient_id="p1", query="hba1c?", status="no_evidence_found")
    no_evidence_answer = enforce_pre_generation_safety(no_evidence_context)

    evidence_context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[_evidence()])
    uncited_answer = build_grounded_answer(evidence_context, "no citation here")

    assert no_evidence_answer.status == uncited_answer.status == "insufficient_evidence"
    assert no_evidence_answer.retrieval_status != uncited_answer.retrieval_status
    assert no_evidence_answer.retrieval_status == "no_evidence_found"
    assert uncited_answer.retrieval_status == "evidence_found"
