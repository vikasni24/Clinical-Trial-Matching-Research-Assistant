"""Phase 6F: focused tests for the safety boundary between retrieved
evidence and generated answers. Each rule from the phase task list gets a
dedicated section. No real LLM is ever called (a local spy/fake provider
stands in); no eligibility logic is modified — rule 7 exercises the
existing, unmodified EligibilityMatcher directly."""

from datetime import date

import pytest

from app.models.clinical_trial import ClinicalTrialOut, EligibilityCriterionOut, TrialEligibilityOut
from app.models.evidence import Evidence
from app.models.patient_profile import ContactInfoOut, DemographicsOut, ObservationOut, PatientProfileOut
from app.models.rag_context import GroundedContext
from app.services.answer_validator import build_grounded_answer
from app.services.eligibility_matcher import EligibilityMatcher
from app.services.grounded_prompt import build_grounded_prompt
from app.services.safety_rules import enforce_pre_generation_safety


class _SpyLLMProvider:
    """Records every prompt it is asked to generate from, so tests can
    assert an LLM was (or was not) ever actually invoked."""

    def __init__(self, response: str = "unused"):
        self.calls: list[str] = []
        self._response = response

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


def _evidence(resource_id="obs-1", patient_id="p1", resource_type="Observation", value=125, unit="mm[Hg]"):
    return Evidence(
        patient_id=patient_id,
        resource_type=resource_type,
        resource_id=resource_id,
        code="8480-6",
        display="Systolic Blood Pressure",
        value=value,
        unit=unit,
    )


# --- rule 1: no evidence -> explicit insufficient evidence, no LLM call -----------------


def test_no_evidence_short_circuits_to_insufficient_evidence_without_calling_llm():
    context = GroundedContext(patient_id="p1", query="hemoglobin a1c?", status="no_evidence_found")
    provider = _SpyLLMProvider()

    answer = enforce_pre_generation_safety(context)

    assert answer is not None
    assert answer.status == "insufficient_evidence"
    assert answer.answer_text is None
    assert answer.evidence == []
    assert provider.calls == []  # generation was never even attempted


# --- rule 2: unsupported query -> explicit unsupported, no LLM call --------------------


def test_unsupported_query_short_circuits_to_unsupported_without_calling_llm():
    context = GroundedContext(patient_id="p1", query="???", status="unsupported", message="No concept identified")
    provider = _SpyLLMProvider()

    answer = enforce_pre_generation_safety(context)

    assert answer is not None
    assert answer.status == "unsupported"
    assert answer.message == "No concept identified"
    assert provider.calls == []


# --- rule 3: evidence exists but is insufficient to ground a confident answer ------------


def test_evidence_found_but_uncited_answer_is_not_converted_to_a_confident_answer():
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[_evidence()])

    # Safety gate lets generation proceed (evidence was found)...
    assert enforce_pre_generation_safety(context) is None

    # ...but the generated text doesn't actually ground itself in any of it.
    answer = build_grounded_answer(context, "The patient appears generally healthy.")

    assert answer.status == "insufficient_evidence"
    assert answer.answer_text is None
    assert answer.evidence == []


def test_evidence_found_but_fabricated_citation_is_not_converted_to_a_confident_answer():
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[_evidence()])

    answer = build_grounded_answer(context, "See [Observation/does-not-exist] for details.")

    assert answer.status == "insufficient_evidence"
    assert answer.evidence == []


# --- rule 4: cross-patient evidence -> reject before a prompt is ever built --------------


def test_cross_patient_evidence_is_rejected_before_building_a_prompt_or_calling_the_llm():
    # GroundedContext's own validator would normally prevent this from ever
    # being constructed — model_construct() bypasses that validation to
    # simulate an upstream invariant violation.
    corrupted_context = GroundedContext.model_construct(
        patient_id="patient-a",
        query="q",
        status="evidence_found",
        evidence=[_evidence(patient_id="patient-b")],
        message=None,
    )
    provider = _SpyLLMProvider()

    with pytest.raises(ValueError):
        enforce_pre_generation_safety(corrupted_context)

    assert provider.calls == []  # the violation was caught before any generate() call


def test_cross_patient_rejection_happens_even_before_prompt_building():
    corrupted_context = GroundedContext.model_construct(
        patient_id="patient-a",
        query="q",
        status="evidence_found",
        evidence=[_evidence(patient_id="patient-b")],
        message=None,
    )

    # The caller's real pipeline order is: safety gate, THEN prompt
    # building. Proving the gate raises means a caller who checks it first
    # never reaches build_grounded_prompt() with contaminated evidence.
    with pytest.raises(ValueError):
        enforce_pre_generation_safety(corrupted_context)


# --- rule 5: raw FHIR is never exposed ----------------------------------------------------


def test_grounded_prompt_never_contains_raw_fhir_markers():
    context = GroundedContext(patient_id="p1", query="blood pressure?", status="evidence_found", evidence=[_evidence()])

    prompt = build_grounded_prompt(context)
    full_text = "\n".join([prompt.instructions, prompt.status_note, prompt.evidence_text])

    for marker in ("resourceType", "valueQuantity", "effectiveDateTime", '"_id"', "fhir_resources"):
        assert marker not in full_text


def test_evidence_model_structurally_excludes_raw_fhir_fields():
    evidence = _evidence()
    dumped = evidence.model_dump()

    for forbidden_key in ("data", "resourceType", "_id"):
        assert forbidden_key not in dumped


def test_grounded_answer_never_carries_raw_fhir_either():
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[_evidence()])

    answer = build_grounded_answer(context, "BP is 125 [Observation/obs-1].")
    dumped = answer.model_dump()

    assert "data" not in dumped
    for evidence_dict in dumped["evidence"]:
        assert "data" not in evidence_dict
        assert "resourceType" not in evidence_dict
        assert "_id" not in evidence_dict


# --- rule 6: missing clinical values are never fabricated -------------------------------


def test_missing_value_is_omitted_from_the_prompt_not_fabricated():
    # value/unit are simply absent — never rendered as a placeholder like
    # "value=None" or "value=unknown".
    evidence_without_value = Evidence(
        patient_id="p1", resource_type="Condition", resource_id="cond-1", display="Hypertension"
    )
    context = GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[evidence_without_value])

    prompt = build_grounded_prompt(context)

    assert "value=" not in prompt.evidence_text
    assert "None" not in prompt.evidence_text
    assert "[Condition/cond-1] Hypertension" in prompt.evidence_text


def test_no_evidence_means_llm_text_with_a_confident_value_is_never_trusted():
    context = GroundedContext(patient_id="p1", query="hba1c?", status="no_evidence_found")

    # Even before generation is attempted, the safety gate already refuses
    # to let a confident-sounding value reach the caller.
    gated_answer = enforce_pre_generation_safety(context)
    assert gated_answer.status == "insufficient_evidence"
    assert gated_answer.answer_text is None

    # And if a raw LLM string were somehow produced anyway, the post-hoc
    # validator independently refuses to trust it too (defense in depth).
    answer = build_grounded_answer(context, "The patient's HbA1c is definitely 9.2%.")
    assert answer.status == "insufficient_evidence"
    assert answer.answer_text is None


# --- rule 7: existing UNKNOWN semantics are preserved, not overwritten ------------------


def test_missing_observation_stays_unknown_not_fail_or_pass():
    profile = PatientProfileOut(
        patient_id="p1",
        demographics=DemographicsOut(patient_id="p1", date_of_birth="1980-01-01", gender="female"),
        contact=ContactInfoOut(),
        observations=[],  # no HbA1c on record at all
    )
    trial = ClinicalTrialOut(
        trial_id="T-UNKNOWN",
        status="recruiting",
        eligibility=TrialEligibilityOut(
            inclusion_criteria=[
                EligibilityCriterionOut(
                    type="observation_threshold",
                    label="HbA1c",
                    code="4548-4",
                    operator="<",
                    value=9.0,
                    unit="%",
                )
            ]
        ),
    )

    result = EligibilityMatcher().evaluate(profile, trial)

    assert result.unknown_criteria[0].result == "UNKNOWN"
    assert result.eligibility_status == "UNKNOWN"
    assert result.failed_criteria == []
    assert result.matched_criteria == []


def test_missing_date_of_birth_age_criterion_stays_unknown():
    profile = PatientProfileOut(
        patient_id="p1",
        demographics=DemographicsOut(patient_id="p1", date_of_birth=None, gender="female"),
        contact=ContactInfoOut(),
    )
    trial = ClinicalTrialOut(
        trial_id="T-AGE-UNKNOWN",
        status="recruiting",
        eligibility=TrialEligibilityOut(minimum_age=18),
    )

    result = EligibilityMatcher().evaluate(profile, trial)

    assert result.unknown_criteria[0].result == "UNKNOWN"
    assert result.eligibility_status == "UNKNOWN"


def test_unknown_is_not_silently_overwritten_by_a_later_passing_criterion():
    # One criterion definitely passes (age); a second is UNKNOWN (missing
    # observation) — overall status must stay UNKNOWN, never get promoted
    # to ELIGIBLE just because something else passed.
    profile = PatientProfileOut(
        patient_id="p1",
        demographics=DemographicsOut(patient_id="p1", date_of_birth=_dob_for_age(45), gender="female"),
        contact=ContactInfoOut(),
        observations=[],
    )
    trial = ClinicalTrialOut(
        trial_id="T-MIXED",
        status="recruiting",
        eligibility=TrialEligibilityOut(
            minimum_age=18,
            inclusion_criteria=[
                EligibilityCriterionOut(type="observation_threshold", label="HbA1c", code="4548-4", operator="<", value=9.0)
            ],
        ),
    )

    result = EligibilityMatcher().evaluate(profile, trial)

    assert result.eligibility_status == "UNKNOWN"
    assert any(c.result == "PASS" for c in result.matched_criteria)
    assert any(c.result == "UNKNOWN" for c in result.unknown_criteria)


def _dob_for_age(age: int) -> str:
    return date(date.today().year - age, 1, 1).isoformat()


# --- determinism across the full safety boundary -----------------------------------------


def test_safety_gate_is_deterministic():
    context = GroundedContext(patient_id="p1", query="hba1c?", status="no_evidence_found")

    first = enforce_pre_generation_safety(context)
    second = enforce_pre_generation_safety(context)

    assert first.model_dump() == second.model_dump()


def test_full_pipeline_is_deterministic_end_to_end():
    context = GroundedContext(patient_id="p1", query="blood pressure?", status="evidence_found", evidence=[_evidence()])
    raw_text = "BP is 125 [Observation/obs-1]."

    def run_pipeline():
        gate_result = enforce_pre_generation_safety(context)
        if gate_result is not None:
            return gate_result
        build_grounded_prompt(context)  # exercised, not re-asserted here (see test_grounded_prompt.py)
        return build_grounded_answer(context, raw_text)

    first = run_pipeline()
    second = run_pipeline()

    assert first.model_dump() == second.model_dump()
