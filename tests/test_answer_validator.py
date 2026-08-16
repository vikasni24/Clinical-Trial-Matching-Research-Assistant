"""Phase 6E: build_grounded_answer — deterministic validation of raw LLM
text against the Evidence actually supplied. No LLM call in this module;
`raw_answer_text` is always supplied directly by the test."""

from app.models.answer import GroundedAnswer
from app.models.evidence import Evidence
from app.models.rag_context import GroundedContext
from app.services.answer_validator import build_grounded_answer


def _evidence(resource_id="obs-1", patient_id="p1", resource_type="Observation"):
    return Evidence(
        patient_id=patient_id,
        resource_type=resource_type,
        resource_id=resource_id,
        code="8480-6",
        display="Systolic Blood Pressure",
        value=125,
        unit="mm[Hg]",
    )


# --- grounded answer -----------------------------------------------------------------


def test_grounded_answer_with_valid_citation():
    context = GroundedContext(patient_id="p1", query="blood pressure?", status="evidence_found", evidence=[_evidence()])

    answer = build_grounded_answer(context, "The patient's systolic BP was 125 mm[Hg] [Observation/obs-1].")

    assert isinstance(answer, GroundedAnswer)
    assert answer.status == "answered"
    assert answer.answer_text == "The patient's systolic BP was 125 mm[Hg] [Observation/obs-1]."
    assert len(answer.evidence) == 1
    assert answer.evidence[0].resource_id == "obs-1"


def test_grounded_answer_retains_all_required_fields():
    context = GroundedContext(patient_id="p1", query="blood pressure?", status="evidence_found", evidence=[_evidence()])

    answer = build_grounded_answer(context, "BP is 125 [Observation/obs-1].")

    assert answer.patient_id == "p1"
    assert answer.query == "blood pressure?"
    assert answer.answer_text is not None
    assert answer.evidence
    assert answer.status == "answered"


def test_evidence_references_resolve_to_supplied_evidence_objects():
    supplied = _evidence()
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[supplied])

    answer = build_grounded_answer(context, "Answer text [Observation/obs-1].")

    # The exact same Evidence object supplied to the context, not a copy
    # or a newly constructed one.
    assert answer.evidence[0] is supplied


def test_only_cited_evidence_is_attached_not_the_full_supplied_set():
    cited = _evidence(resource_id="obs-cited")
    uncited = _evidence(resource_id="obs-uncited")
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[cited, uncited])

    answer = build_grounded_answer(context, "Only this matters [Observation/obs-cited].")

    assert [e.resource_id for e in answer.evidence] == ["obs-cited"]


def test_multiple_valid_citations_are_all_attached():
    e1 = _evidence(resource_id="obs-1")
    e2 = _evidence(resource_id="obs-2")
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[e1, e2])

    answer = build_grounded_answer(context, "See [Observation/obs-1] and [Observation/obs-2].")

    assert answer.status == "answered"
    assert {e.resource_id for e in answer.evidence} == {"obs-1", "obs-2"}


# --- unsupported answer ----------------------------------------------------------------


def test_unsupported_context_produces_unsupported_answer():
    context = GroundedContext(patient_id="p1", query="???", status="unsupported", message="No concept identified")

    answer = build_grounded_answer(context, "This text is never trusted or inspected.")

    assert answer.status == "unsupported"
    assert answer.answer_text is None
    assert answer.evidence == []
    assert answer.message == "No concept identified"


# --- missing evidence -------------------------------------------------------------------


def test_no_evidence_context_never_trusts_the_raw_answer_text():
    context = GroundedContext(patient_id="p1", query="hemoglobin a1c?", status="no_evidence_found")

    # Even a confident-sounding LLM answer must be discarded when no
    # evidence was ever available to ground it.
    answer = build_grounded_answer(context, "The patient's HbA1c is definitely 9.2%.")

    assert answer.status == "insufficient_evidence"
    assert answer.answer_text is None
    assert answer.evidence == []


def test_uncited_answer_becomes_insufficient_evidence_even_with_evidence_available():
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[_evidence()])

    answer = build_grounded_answer(context, "The patient appears generally healthy.")

    assert answer.status == "insufficient_evidence"
    assert answer.answer_text is None
    assert answer.evidence == []
    assert "did not cite" in answer.message.lower()


# --- invalid evidence reference / fabricated resource reference -------------------------


def test_invalid_evidence_reference_is_rejected():
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[_evidence(resource_id="obs-1")])

    # References a resource_id that was never supplied.
    answer = build_grounded_answer(context, "See [Observation/obs-does-not-exist].")

    assert answer.status == "insufficient_evidence"
    assert answer.answer_text is None
    assert answer.evidence == []
    assert "fabricated" in answer.message.lower()


def test_fabricated_resource_reference_of_a_different_type_is_rejected():
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[_evidence(resource_type="Observation")])

    # Fabricates an entirely different resource_type/id pair.
    answer = build_grounded_answer(context, "Patient has [Condition/fake-condition-id].")

    assert answer.status == "insufficient_evidence"
    assert answer.evidence == []


def test_mixed_valid_and_fabricated_citations_reject_the_whole_answer():
    real = _evidence(resource_id="obs-real")
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[real])

    # One real citation alongside one fabricated one — the presence of any
    # fabrication is disqualifying, not silently dropped in favor of the
    # real one.
    answer = build_grounded_answer(context, "See [Observation/obs-real] and also [Observation/obs-fake].")

    assert answer.status == "insufficient_evidence"
    assert answer.evidence == []


# --- cross-patient evidence (defense in depth) -------------------------------------------


def test_cross_patient_evidence_raises_value_error():
    # GroundedContext's own validator would normally prevent this from
    # ever being constructed — model_construct() deliberately bypasses
    # that validation to simulate an upstream invariant violation and
    # prove this layer independently defends against it too.
    other_patient_evidence = _evidence(resource_id="obs-1", patient_id="patient-b")
    corrupted_context = GroundedContext.model_construct(
        patient_id="patient-a", query="q", status="evidence_found", evidence=[other_patient_evidence], message=None
    )

    try:
        build_grounded_answer(corrupted_context, "Answer [Observation/obs-1].")
        raised = False
    except ValueError:
        raised = True

    assert raised


# --- deterministic validation --------------------------------------------------------------


def test_deterministic_repeated_validation():
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[_evidence()])
    raw_text = "BP is 125 [Observation/obs-1]."

    first = build_grounded_answer(context, raw_text)
    second = build_grounded_answer(context, raw_text)

    assert first.model_dump() == second.model_dump()


def test_deterministic_rejection_is_also_repeatable():
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[_evidence()])
    raw_text = "See [Observation/does-not-exist]."

    first = build_grounded_answer(context, raw_text)
    second = build_grounded_answer(context, raw_text)

    assert first.model_dump() == second.model_dump()
    assert first.status == "insufficient_evidence"


# --- no fabrication of resource IDs / clinical values / evidence ---------------------------


def test_no_evidence_object_is_ever_fabricated():
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[_evidence()])

    answer = build_grounded_answer(context, "Nonsense [Observation/nope] [Condition/also-nope].")

    # Rejected entirely — never partially trusted with invented placeholders.
    assert answer.evidence == []
    assert answer.status == "insufficient_evidence"
