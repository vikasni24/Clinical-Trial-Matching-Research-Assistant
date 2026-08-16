"""Deterministic three-state (PASS/FAIL/UNKNOWN) eligibility evaluation of a
PatientProfile against a ClinicalTrial. No LLMs, no embeddings, no
probabilistic inference — every result is derived directly from explicit
fields already present in the profile and the trial definition.

SCORING METHODOLOGY (documented per Phase 3 Step 9):
    match_score = (number of criteria that evaluated PASS) / (total criteria
    evaluated), rounded to 2 decimals. It is a transparent count of how many
    of the trial's defined, evaluable criteria the patient's *known* data
    satisfies — it is NOT a probability of eligibility, enrollment, or
    trial success, and must never be presented as one.

THREE-STATE RULE:
    A criterion is UNKNOWN whenever the PatientProfile lacks the data needed
    to evaluate it (e.g. no recorded observation of the required type, no
    date_of_birth). A criterion is never forced to PASS or FAIL when the
    necessary patient data is simply absent.

OVERALL ELIGIBILITY RULE (deterministic, evaluated in this precedence):
    1. Any FAILed criterion  -> INELIGIBLE (a definite disqualifying fact)
    2. Else any UNKNOWN criterion -> UNKNOWN (missing data, no confident call)
    3. Else (every criterion PASSed) -> ELIGIBLE

PRESENCE ASSUMPTION: PatientProfile.conditions/medications/allergies are
built from a patient's *complete* ingested resource history (Phase 2), so
"no matching condition/medication/allergy in the list" is treated as a
confident FAIL (for inclusion) — not UNKNOWN. Observations are different:
a lab test may simply never have been ordered, so a missing observation of
the required type is UNKNOWN, matching the Phase 3 spec's own example
(missing HbA1c -> UNKNOWN, not INELIGIBLE).

EVIDENCE (Phase 4D): the PASS/FAIL/UNKNOWN decision itself is still made
entirely from the already-normalized PatientProfile, exactly as above —
that logic is unchanged. Separately, when a criterion's decision was
backed by a specific FHIR resource (a matched Condition/Observation/
MedicationRequest/AllergyIntolerance, or the Patient resource for
age/sex), an optional EvidenceService is used to fetch a traceable
Evidence record for it via a single targeted, patient-scoped lookup
(EvidenceService -> EvidenceRepository -> MongoDB `fhir_resources`) and
attach it to that CriterionEvaluationOut. The matcher never queries
MongoDB itself. If no EvidenceService is supplied (the default), the
matcher behaves exactly as before Phase 4D — evidence lists are simply
empty; nothing else changes.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Optional

from app.models.clinical_trial import ClinicalTrialOut, EligibilityCriterionOut
from app.models.evidence import Evidence
from app.models.match_result import CriterionEvaluationOut, MatchResultOut
from app.models.patient_profile import PatientProfileOut
from app.services.evidence_service import EvidenceService

# Which FHIR resource_type backs each structured criterion type — used only
# to look up traceability evidence for a criterion that already matched;
# never used to make the PASS/FAIL/UNKNOWN decision itself.
_RESOURCE_TYPE_BY_CRITERION_TYPE = {
    "condition": "Condition",
    "medication": "MedicationRequest",
    "allergy": "AllergyIntolerance",
    "observation_threshold": "Observation",
}


def _fetch_evidence(
    patient_id: str,
    resource_type: Optional[str],
    resource_id: Optional[str],
    evidence_service: Optional[EvidenceService],
) -> list[Evidence]:
    """A single targeted, patient-scoped lookup for traceability — never a
    scan, never touches MongoDB directly (goes through EvidenceService).
    Empty whenever there's no evidence_service, or no concrete resource
    backing this evaluation (e.g. UNKNOWN, or a FAIL because nothing
    matched)."""
    if evidence_service is None or resource_type is None or resource_id is None:
        return []
    evidence = evidence_service.get_patient_resource_evidence(patient_id, resource_type, resource_id)
    return [evidence] if evidence is not None else []

_OPERATORS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
}


def _calculate_age(date_of_birth: Optional[str]) -> Optional[int]:
    if not date_of_birth:
        return None
    try:
        year, month, day = (int(part) for part in date_of_birth.split("-")[:3])
    except (ValueError, TypeError):
        return None
    today = date.today()
    return today.year - year - ((today.month, today.day) < (month, day))


def _age_criterion(
    age: Optional[int],
    symbol: str,
    threshold: int,
    comparator,
    patient_id: str,
    evidence_service: Optional[EvidenceService],
) -> CriterionEvaluationOut:
    criterion_label = f"Age {symbol} {threshold}"
    if age is None:
        return CriterionEvaluationOut(
            criterion=criterion_label,
            category="demographics",
            requirement="inclusion",
            result="UNKNOWN",
            patient_value=None,
            required_value=f"{symbol} {threshold}",
            reason="Patient date of birth is not recorded",
        )
    # Age is derived from the Patient resource's birthDate, so that resource
    # is the evidence backing this evaluation whenever age is known.
    evidence = _fetch_evidence(patient_id, "Patient", patient_id, evidence_service)
    return CriterionEvaluationOut(
        criterion=criterion_label,
        category="demographics",
        requirement="inclusion",
        result="PASS" if comparator(age, threshold) else "FAIL",
        patient_value=age,
        required_value=f"{symbol} {threshold}",
        reason=f"Patient age is {age}",
        evidence=evidence,
    )


def _evaluate_age(
    profile: PatientProfileOut,
    trial: ClinicalTrialOut,
    patient_id: str,
    evidence_service: Optional[EvidenceService],
) -> list[CriterionEvaluationOut]:
    eligibility = trial.eligibility
    age = _calculate_age(profile.demographics.date_of_birth)

    evaluations: list[CriterionEvaluationOut] = []
    if eligibility.minimum_age is not None:
        evaluations.append(_age_criterion(age, ">=", eligibility.minimum_age, _OPERATORS[">="], patient_id, evidence_service))
    if eligibility.maximum_age is not None:
        evaluations.append(_age_criterion(age, "<=", eligibility.maximum_age, _OPERATORS["<="], patient_id, evidence_service))
    return evaluations


def _evaluate_sex(
    profile: PatientProfileOut,
    trial: ClinicalTrialOut,
    patient_id: str,
    evidence_service: Optional[EvidenceService],
) -> Optional[CriterionEvaluationOut]:
    required_sex = trial.eligibility.sex
    if not required_sex or required_sex.lower() == "all":
        return None

    gender = profile.demographics.gender
    if not gender:
        return CriterionEvaluationOut(
            criterion=f"Sex == {required_sex}",
            category="demographics",
            requirement="inclusion",
            result="UNKNOWN",
            patient_value=None,
            required_value=required_sex,
            reason="Patient sex/gender is not recorded",
        )

    passed = gender.lower() == required_sex.lower()
    # Gender is derived from the Patient resource, so that resource is the
    # evidence backing this evaluation whenever gender is known.
    evidence = _fetch_evidence(patient_id, "Patient", patient_id, evidence_service)
    return CriterionEvaluationOut(
        criterion=f"Sex == {required_sex}",
        category="demographics",
        requirement="inclusion",
        result="PASS" if passed else "FAIL",
        patient_value=gender,
        required_value=required_sex,
        reason=f"Patient sex is '{gender}'",
        evidence=evidence,
    )


def _matches_coded_item(item: Any, criterion: EligibilityCriterionOut, display_attr: str) -> bool:
    """A patient item (condition/medication/allergy) matches a criterion if
    their codes agree, or — when either side lacks a code — their display
    text matches exactly, case-insensitively. Deterministic, no fuzzy/NLP
    matching."""
    item_code = getattr(item, "code", None)
    if criterion.code and item_code:
        return item_code == criterion.code
    item_display = getattr(item, display_attr, None)
    if criterion.display and item_display:
        return item_display.strip().lower() == criterion.display.strip().lower()
    return False


def _find_matching_item(items: list[Any], criterion: EligibilityCriterionOut, display_attr: str) -> Optional[Any]:
    for item in items:
        if _matches_coded_item(item, criterion, display_attr):
            return item
    return None


def _condition_met(profile: PatientProfileOut, criterion: EligibilityCriterionOut) -> tuple[Optional[bool], Optional[Any]]:
    """Returns (condition_met, matched_item).

    condition_met is True/False (the underlying fact is present/absent in
    the patient's complete recorded data) or None (cannot be determined) —
    this is the exact same decision as before Phase 4D, unchanged.

    matched_item is the specific ConditionOut/MedicationOut/AllergyOut/
    ObservationOut that was found (carrying its resource_id, for evidence
    traceability), or None when nothing was found/matched.

    A condition/medication/allergy criterion with neither a `code` nor a
    `display` carries no identifying information at all — it can never be
    matched against anything, so it is UNKNOWN rather than a confident FAIL.
    This is distinct from a well-specified criterion the patient simply
    doesn't have on record, which IS a confident FAIL (see module docstring)."""
    if criterion.type == "condition":
        if not criterion.code and not criterion.display:
            return None, None
        item = _find_matching_item(profile.conditions, criterion, "display")
        return item is not None, item
    if criterion.type == "medication":
        if not criterion.code and not criterion.display:
            return None, None
        item = _find_matching_item(profile.medications, criterion, "medication_name")
        return item is not None, item
    if criterion.type == "allergy":
        if not criterion.code and not criterion.display:
            return None, None
        item = _find_matching_item(profile.allergies, criterion, "substance")
        return item is not None, item
    if criterion.type == "observation_threshold":
        met = _observation_threshold_met(profile, criterion)
        item = _latest_matching_observation(profile, criterion)
        return met, item
    return None, None


def _latest_matching_observation(profile: PatientProfileOut, criterion: EligibilityCriterionOut) -> Optional[Any]:
    """The most recent (by effective_date) patient observation matching this
    criterion's code, or display text if no code is given. None if the
    patient has no such observation on record."""
    matching = [
        obs
        for obs in profile.observations
        if (criterion.code and obs.code == criterion.code)
        or (not criterion.code and criterion.display and obs.name and obs.name.strip().lower() == criterion.display.strip().lower())
    ]
    if not matching:
        return None
    # Deterministic tie-break: most recent effective_date wins; missing dates sort last.
    matching.sort(key=lambda o: (o.effective_date or ""), reverse=True)
    return matching[0]


def _observation_threshold_met(profile: PatientProfileOut, criterion: EligibilityCriterionOut) -> Optional[bool]:
    latest = _latest_matching_observation(profile, criterion)
    if latest is None:
        return None

    if not isinstance(latest.value, (int, float)) or criterion.value is None or not criterion.operator:
        return None

    comparator = _OPERATORS.get(criterion.operator)
    if comparator is None:
        return None
    return comparator(latest.value, criterion.value)


def _requirement_result(condition_met: Optional[bool], requirement: str) -> str:
    """Maps the underlying fact (present/absent/unknown) to PASS/FAIL/UNKNOWN
    given whether this criterion is an inclusion or exclusion requirement."""
    if condition_met is None:
        return "UNKNOWN"
    if requirement == "exclusion":
        return "FAIL" if condition_met else "PASS"
    return "PASS" if condition_met else "FAIL"


_CATEGORY_BY_TYPE = {
    "condition": "condition",
    "medication": "medication",
    "allergy": "allergy",
    "observation_threshold": "observation",
}


def _evaluate_structured_criterion(
    profile: PatientProfileOut,
    criterion: EligibilityCriterionOut,
    requirement: str,
    patient_id: str,
    evidence_service: Optional[EvidenceService],
) -> CriterionEvaluationOut:
    condition_met, matched_item = _condition_met(profile, criterion)
    result = _requirement_result(condition_met, requirement)
    category = _CATEGORY_BY_TYPE.get(criterion.type, criterion.type)
    resource_type = _RESOURCE_TYPE_BY_CRITERION_TYPE.get(criterion.type)
    resource_id = getattr(matched_item, "resource_id", None)
    evidence = _fetch_evidence(patient_id, resource_type, resource_id, evidence_service)

    if criterion.type == "observation_threshold":
        required_value = f"{criterion.operator} {criterion.value}" + (f" {criterion.unit}" if criterion.unit else "")
        value = matched_item.value if matched_item else None
        unit = matched_item.unit if matched_item else None
        if condition_met is None:
            reason = f"No '{criterion.label}' observation is available for this patient"
        else:
            reason = f"Patient's most recent '{criterion.label}' is {value}{f' {unit}' if unit else ''}"
        return CriterionEvaluationOut(
            criterion=criterion.label,
            category=category,
            requirement=requirement,
            result=result,
            patient_value=value,
            required_value=required_value,
            reason=reason,
            evidence=evidence,
        )

    presence_noun = {"condition": "condition", "medication": "medication", "allergy": "allergy"}.get(
        criterion.type, criterion.type
    )
    if condition_met is None:
        reason = f"Cannot determine whether patient has a recorded {presence_noun} matching '{criterion.label}'"
        patient_value = None
    elif condition_met:
        reason = f"Patient has a recorded {presence_noun} matching '{criterion.label}'"
        patient_value = True
    else:
        reason = f"No recorded {presence_noun} matches '{criterion.label}'"
        patient_value = False

    required_value = "present" if requirement == "inclusion" else "absent"
    return CriterionEvaluationOut(
        criterion=criterion.label,
        category=category,
        requirement=requirement,
        result=result,
        patient_value=patient_value,
        required_value=required_value,
        reason=reason,
        evidence=evidence,
    )


def _build_explanation(
    eligibility_status: str,
    matched: list[CriterionEvaluationOut],
    failed: list[CriterionEvaluationOut],
    unknown: list[CriterionEvaluationOut],
    total: int,
) -> str:
    parts = [f"{len(matched)} of {total} criteria matched."]
    if failed:
        failed_labels = ", ".join(c.criterion for c in failed)
        parts.append(f"{eligibility_status} due to {len(failed)} failed criterion/criteria: {failed_labels}.")
    elif unknown:
        unknown_labels = ", ".join(c.criterion for c in unknown)
        parts.append(f"{eligibility_status}: {len(unknown)} criterion/criteria could not be determined: {unknown_labels}.")
    else:
        parts.append(f"{eligibility_status}: all criteria satisfied.")
    return " ".join(parts)


class EligibilityMatcher:
    """Deterministic evaluator: same PatientProfile + same ClinicalTrial +
    same MongoDB state always produces the same MatchResult.

    evidence_service is optional. When supplied (see TrialMatchingService),
    each criterion evaluation that was backed by a specific FHIR resource
    gets a traceable Evidence record attached via a single targeted lookup.
    When omitted (the default), the matcher behaves exactly as it did
    before Phase 4D — decision logic is identical either way."""

    def __init__(self, evidence_service: Optional[EvidenceService] = None):
        self._evidence_service = evidence_service

    def evaluate(self, profile: PatientProfileOut, trial: ClinicalTrialOut) -> MatchResultOut:
        patient_id = profile.patient_id
        evaluations: list[CriterionEvaluationOut] = []

        evaluations.extend(_evaluate_age(profile, trial, patient_id, self._evidence_service))
        sex_evaluation = _evaluate_sex(profile, trial, patient_id, self._evidence_service)
        if sex_evaluation is not None:
            evaluations.append(sex_evaluation)

        for criterion in trial.eligibility.inclusion_criteria:
            evaluations.append(
                _evaluate_structured_criterion(profile, criterion, "inclusion", patient_id, self._evidence_service)
            )
        for criterion in trial.eligibility.exclusion_criteria:
            evaluations.append(
                _evaluate_structured_criterion(profile, criterion, "exclusion", patient_id, self._evidence_service)
            )

        matched = [e for e in evaluations if e.result == "PASS"]
        failed = [e for e in evaluations if e.result == "FAIL"]
        unknown = [e for e in evaluations if e.result == "UNKNOWN"]

        if failed:
            eligibility_status = "INELIGIBLE"
        elif unknown:
            eligibility_status = "UNKNOWN"
        else:
            eligibility_status = "ELIGIBLE"

        total = len(evaluations)
        match_score = round(len(matched) / total, 2) if total else 0.0

        return MatchResultOut(
            patient_id=profile.patient_id,
            trial_id=trial.trial_id,
            overall_status=trial.status,
            eligibility_status=eligibility_status,
            match_score=match_score,
            matched_criteria=matched,
            failed_criteria=failed,
            unknown_criteria=unknown,
            explanation=_build_explanation(eligibility_status, matched, failed, unknown, total),
            evaluated_at=datetime.now(timezone.utc),
        )
