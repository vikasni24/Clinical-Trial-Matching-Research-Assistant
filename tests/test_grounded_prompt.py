"""Phase 6B: deterministic conversion of GroundedContext into a strict,
LLM-ready GroundedPrompt. No LLM call exists anywhere in this module or
these tests."""

import inspect

from app.models.evidence import Evidence
from app.models.rag_context import GroundedContext
from app.services.grounded_prompt import GroundedPrompt, SAFETY_INSTRUCTIONS, build_grounded_prompt


def _evidence(patient_id="p1", resource_id="obs-1"):
    return Evidence(
        patient_id=patient_id,
        resource_type="Observation",
        resource_id=resource_id,
        code="8480-6",
        display="Systolic Blood Pressure",
        value=125,
        unit="mm[Hg]",
        effective_date="2023-01-10",
        status="final",
    )


# --- evidence formatting ------------------------------------------------------------


def test_evidence_formatting_is_traceable_and_flat():
    context = GroundedContext(patient_id="p1", query="blood pressure?", status="evidence_found", evidence=[_evidence()])

    prompt = build_grounded_prompt(context)

    assert "[Observation/obs-1]" in prompt.evidence_text
    assert "Systolic Blood Pressure" in prompt.evidence_text
    assert "code=8480-6" in prompt.evidence_text
    assert "value=125 mm[Hg]" in prompt.evidence_text
    assert "date=2023-01-10" in prompt.evidence_text
    assert "status=final" in prompt.evidence_text


def test_evidence_formatting_handles_multiple_items_deterministically():
    e1 = _evidence(resource_id="obs-1")
    e2 = _evidence(resource_id="obs-2")
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[e1, e2])

    prompt = build_grounded_prompt(context)

    lines = prompt.evidence_text.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("[Observation/obs-1]")
    assert lines[1].startswith("[Observation/obs-2]")


def test_evidence_formatting_handles_sparse_evidence_gracefully():
    sparse = Evidence(patient_id="p1", resource_type="Patient", resource_id="p1")  # only traceability fields
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[sparse])

    prompt = build_grounded_prompt(context)

    assert "[Patient/p1]" in prompt.evidence_text
    assert "(no additional detail recorded)" in prompt.evidence_text


# --- no-evidence / unsupported behavior ----------------------------------------------


def test_no_evidence_behavior():
    context = GroundedContext(patient_id="p1", query="hemoglobin a1c?", status="no_evidence_found")

    prompt = build_grounded_prompt(context)

    assert prompt.status == "no_evidence_found"
    assert prompt.evidence == []
    assert "No evidence was supplied" in prompt.evidence_text
    assert "insufficient" in prompt.status_note.lower()
    assert "never guess" in prompt.status_note.lower()


def test_unsupported_behavior():
    context = GroundedContext(patient_id="p1", query="???", status="unsupported", message="No concept identified")

    prompt = build_grounded_prompt(context)

    assert prompt.status == "unsupported"
    assert prompt.evidence == []
    assert "could not be processed" in prompt.status_note.lower()
    assert "never guess" in prompt.status_note.lower()


def test_no_evidence_and_unsupported_notes_are_distinct():
    no_evidence = build_grounded_prompt(GroundedContext(patient_id="p1", query="q", status="no_evidence_found"))
    unsupported = build_grounded_prompt(GroundedContext(patient_id="p1", query="q", status="unsupported"))

    assert no_evidence.status_note != unsupported.status_note


# --- patient isolation -----------------------------------------------------------------


def test_patient_isolation_preserved_in_prompt():
    context = GroundedContext(patient_id="patient-a", query="q", status="evidence_found", evidence=[_evidence(patient_id="patient-a")])

    prompt = build_grounded_prompt(context)

    assert prompt.patient_id == "patient-a"
    assert all(e.patient_id == "patient-a" for e in prompt.evidence)


def test_prompt_instructs_never_using_other_patient_evidence():
    assert "different patient" not in SAFETY_INSTRUCTIONS  # sanity: not accidentally naming a specific patient
    assert "Never use or reference evidence belonging to a patient other than the one specified" in SAFETY_INSTRUCTIONS


# --- raw FHIR exclusion -----------------------------------------------------------------


def test_no_raw_fhir_or_mongo_id_in_prompt():
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[_evidence()])

    prompt = build_grounded_prompt(context)
    dumped = prompt.model_dump()

    assert "data" not in dumped
    assert "resourceType" not in prompt.evidence_text
    assert "_id" not in prompt.evidence_text
    for evidence_dict in dumped["evidence"]:
        assert "data" not in evidence_dict
        assert "resourceType" not in evidence_dict
        assert "_id" not in evidence_dict


# --- determinism -------------------------------------------------------------------------


def test_deterministic_repeated_output():
    context = GroundedContext(patient_id="p1", query="blood pressure?", status="evidence_found", evidence=[_evidence()])

    first = build_grounded_prompt(context)
    second = build_grounded_prompt(context)

    assert first.model_dump() == second.model_dump()


# --- explicit required directives ---------------------------------------------------------


def test_instructions_cover_all_nine_required_directives():
    required_phrase_fragments = [
        "Use ONLY the evidence supplied",  # 1. evidence-only grounding
        "Do not invent, guess, or infer",  # 2. no invented facts
        "Absence of evidence is not evidence of absence",  # 3. missing != negative fact
        "say so explicitly rather than guessing",  # 4. explicit insufficiency
        "Do not describe, quote, or expose raw FHIR documents",  # 5. no raw FHIR
        "traceable to a specific evidence item by its resource_type and resource_id",  # 6. traceability
        "Never use or reference evidence belonging to a patient other than the one specified",  # 7. patient isolation
        "Do not fabricate citations, resource IDs",  # 8. no fabricated citations
        "Do not state or imply certainty beyond what the supplied evidence actually supports",  # 9. no unsupported certainty
    ]
    for fragment in required_phrase_fragments:
        assert fragment in SAFETY_INSTRUCTIONS


def test_instructions_are_identical_regardless_of_context():
    # The safety instructions are a fixed template — never rewritten per query/patient.
    a = build_grounded_prompt(GroundedContext(patient_id="p1", query="q1", status="evidence_found", evidence=[_evidence()]))
    b = build_grounded_prompt(GroundedContext(patient_id="p2", query="q2", status="no_evidence_found"))

    assert a.instructions == b.instructions == SAFETY_INSTRUCTIONS


# --- evidence traceability -----------------------------------------------------------------


def test_every_evidence_item_is_individually_citeable():
    items = [_evidence(resource_id=f"obs-{i}") for i in range(3)]
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=items)

    prompt = build_grounded_prompt(context)

    for item in items:
        assert f"[{item.resource_type}/{item.resource_id}]" in prompt.evidence_text
    # And the structured list is available too, not just the text rendering.
    assert prompt.evidence == items


def test_prompt_evidence_field_is_the_same_evidence_objects():
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[_evidence()])

    prompt = build_grounded_prompt(context)

    assert prompt.evidence == context.evidence


# --- no LLM functionality -------------------------------------------------------------------


def test_no_llm_functionality_exists():
    import app.services.grounded_prompt as module

    imports = [
        line.strip()
        for line in inspect.getsource(module).splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    import_text = "\n".join(imports).lower()
    for forbidden in ("openai", "anthropic", "langchain", "llamaindex", "requests", "httpx"):
        assert forbidden not in import_text


def test_grounded_prompt_has_no_scoring_or_reasoning_fields():
    prohibited = {"confidence_score", "hallucination_score", "ai_reasoning", "chain_of_thought", "probability"}
    assert prohibited.isdisjoint(GroundedPrompt.model_fields.keys())
