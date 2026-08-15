"""Unit tests for the deterministic EligibilityMatcher — pure in-memory
PatientProfile/ClinicalTrial objects, no MongoDB."""

from datetime import date

from app.models.clinical_trial import ClinicalTrialOut, EligibilityCriterionOut, TrialEligibilityOut
from app.models.patient_profile import (
    AllergyOut,
    ConditionOut,
    ContactInfoOut,
    DemographicsOut,
    MedicationOut,
    ObservationOut,
    PatientProfileOut,
)
from app.services.eligibility_matcher import EligibilityMatcher

matcher = EligibilityMatcher()


def _dob_for_age(age: int) -> str:
    return date(date.today().year - age, 1, 1).isoformat()


def _profile(date_of_birth=None, gender=None, conditions=None, observations=None, medications=None, allergies=None) -> PatientProfileOut:
    return PatientProfileOut(
        patient_id="p1",
        demographics=DemographicsOut(patient_id="p1", date_of_birth=date_of_birth, gender=gender),
        contact=ContactInfoOut(),
        conditions=conditions or [],
        observations=observations or [],
        medications=medications or [],
        allergies=allergies or [],
    )


def _trial(minimum_age=None, maximum_age=None, sex=None, inclusion_criteria=None, exclusion_criteria=None, status="recruiting") -> ClinicalTrialOut:
    return ClinicalTrialOut(
        trial_id="T1",
        status=status,
        eligibility=TrialEligibilityOut(
            minimum_age=minimum_age,
            maximum_age=maximum_age,
            sex=sex,
            inclusion_criteria=inclusion_criteria or [],
            exclusion_criteria=exclusion_criteria or [],
        ),
    )


# --- age ---------------------------------------------------------------


def test_age_pass():
    result = matcher.evaluate(_profile(date_of_birth=_dob_for_age(45)), _trial(minimum_age=18))
    assert result.matched_criteria[0].result == "PASS"
    assert result.matched_criteria[0].patient_value == 45
    assert result.eligibility_status == "ELIGIBLE"


def test_age_fail():
    result = matcher.evaluate(_profile(date_of_birth=_dob_for_age(15)), _trial(minimum_age=18))
    assert result.failed_criteria[0].result == "FAIL"
    assert result.eligibility_status == "INELIGIBLE"


def test_age_unknown_when_dob_missing():
    result = matcher.evaluate(_profile(date_of_birth=None), _trial(minimum_age=18))
    assert result.unknown_criteria[0].result == "UNKNOWN"
    assert result.eligibility_status == "UNKNOWN"
    assert result.unknown_criteria[0].patient_value is None


def test_age_maximum_boundary_fail():
    result = matcher.evaluate(_profile(date_of_birth=_dob_for_age(70)), _trial(maximum_age=65))
    assert result.eligibility_status == "INELIGIBLE"
    assert result.failed_criteria[0].criterion == "Age <= 65"


# --- sex ---------------------------------------------------------------


def test_sex_pass():
    result = matcher.evaluate(_profile(gender="female"), _trial(sex="female"))
    assert result.eligibility_status == "ELIGIBLE"
    assert result.matched_criteria[0].result == "PASS"


def test_sex_fail():
    result = matcher.evaluate(_profile(gender="male"), _trial(sex="female"))
    assert result.eligibility_status == "INELIGIBLE"
    assert result.failed_criteria[0].patient_value == "male"


def test_sex_unknown_when_gender_missing():
    result = matcher.evaluate(_profile(gender=None), _trial(sex="female"))
    assert result.eligibility_status == "UNKNOWN"
    assert result.unknown_criteria[0].reason == "Patient sex/gender is not recorded"


def test_sex_all_generates_no_criterion():
    result = matcher.evaluate(_profile(gender=None), _trial(sex="all"))
    all_criteria = result.matched_criteria + result.failed_criteria + result.unknown_criteria
    assert not any("Sex" in c.criterion for c in all_criteria)
    assert result.eligibility_status == "ELIGIBLE"  # no criteria at all -> vacuously eligible


# --- condition ---------------------------------------------------------------


def test_condition_pass():
    criterion = EligibilityCriterionOut(type="condition", label="Hypertension", code="59621000", display="Hypertension")
    profile = _profile(conditions=[ConditionOut(resource_id="c1", code="59621000", display="Hypertension")])
    result = matcher.evaluate(profile, _trial(inclusion_criteria=[criterion]))
    assert result.eligibility_status == "ELIGIBLE"
    assert result.matched_criteria[0].patient_value is True


def test_condition_fail_when_absent():
    criterion = EligibilityCriterionOut(type="condition", label="Hypertension", code="59621000", display="Hypertension")
    profile = _profile(conditions=[])
    result = matcher.evaluate(profile, _trial(inclusion_criteria=[criterion]))
    assert result.eligibility_status == "INELIGIBLE"
    assert result.failed_criteria[0].criterion == "Hypertension"
    assert result.failed_criteria[0].patient_value is False


def test_condition_unknown_when_criterion_has_no_identifying_code_or_display():
    # A criterion with neither code nor display can never be matched against
    # anything — that's UNKNOWN (unevaluable), not a confident FAIL.
    criterion = EligibilityCriterionOut(type="condition", label="Some unspecified condition")
    profile = _profile(conditions=[ConditionOut(resource_id="c1", display="Diabetes")])
    result = matcher.evaluate(profile, _trial(inclusion_criteria=[criterion]))
    assert result.unknown_criteria[0].result == "UNKNOWN"
    assert result.eligibility_status == "UNKNOWN"


def test_unrecognized_criterion_type_never_defaults_to_pass():
    criterion = EligibilityCriterionOut(type="something_new", label="Mystery criterion")
    result = matcher.evaluate(_profile(), _trial(inclusion_criteria=[criterion]))
    assert result.unknown_criteria[0].result == "UNKNOWN"
    assert result.eligibility_status == "UNKNOWN"


# --- observation threshold ---------------------------------------------------------------


def test_observation_threshold_pass():
    criterion = EligibilityCriterionOut(type="observation_threshold", label="BMI", code="39156-5", operator=">=", value=30, unit="kg/m2")
    profile = _profile(observations=[ObservationOut(resource_id="o1", code="39156-5", name="Body Mass Index", value=32.5, value_type="Quantity", unit="kg/m2", effective_date="2023-01-01")])
    result = matcher.evaluate(profile, _trial(inclusion_criteria=[criterion]))
    assert result.eligibility_status == "ELIGIBLE"
    assert result.matched_criteria[0].patient_value == 32.5


def test_observation_threshold_fail():
    criterion = EligibilityCriterionOut(type="observation_threshold", label="BMI", code="39156-5", operator=">=", value=30, unit="kg/m2")
    profile = _profile(observations=[ObservationOut(resource_id="o1", code="39156-5", name="Body Mass Index", value=22.0, value_type="Quantity", effective_date="2023-01-01")])
    result = matcher.evaluate(profile, _trial(inclusion_criteria=[criterion]))
    assert result.eligibility_status == "INELIGIBLE"
    assert result.failed_criteria[0].patient_value == 22.0


def test_observation_threshold_unknown_when_no_matching_observation():
    criterion = EligibilityCriterionOut(type="observation_threshold", label="HbA1c", code="4548-4", operator=">=", value=7.0, unit="%")
    profile = _profile(observations=[])
    result = matcher.evaluate(profile, _trial(inclusion_criteria=[criterion]))
    assert result.eligibility_status == "UNKNOWN"
    assert result.unknown_criteria[0].reason == "No 'HbA1c' observation is available for this patient"
    assert result.unknown_criteria[0].patient_value is None


def test_observation_threshold_uses_most_recent_value():
    criterion = EligibilityCriterionOut(type="observation_threshold", label="Glucose", code="2339-0", operator=">=", value=100, unit="mg/dL")
    profile = _profile(observations=[
        ObservationOut(resource_id="o1", code="2339-0", name="Glucose", value=80, value_type="Quantity", effective_date="2020-01-01"),
        ObservationOut(resource_id="o2", code="2339-0", name="Glucose", value=150, value_type="Quantity", effective_date="2023-06-01"),
    ])
    result = matcher.evaluate(profile, _trial(inclusion_criteria=[criterion]))
    assert result.matched_criteria[0].patient_value == 150


# --- medication ---------------------------------------------------------------


def test_medication_required_pass():
    criterion = EligibilityCriterionOut(type="medication", label="Warfarin", code="855332", display="Warfarin Sodium 5 MG Oral Tablet")
    profile = _profile(medications=[MedicationOut(resource_id="m1", code="855332", medication_name="Warfarin Sodium 5 MG Oral Tablet")])
    result = matcher.evaluate(profile, _trial(inclusion_criteria=[criterion]))
    assert result.eligibility_status == "ELIGIBLE"


def test_medication_required_fail_when_absent():
    criterion = EligibilityCriterionOut(type="medication", label="Warfarin", code="855332")
    profile = _profile(medications=[])
    result = matcher.evaluate(profile, _trial(inclusion_criteria=[criterion]))
    assert result.eligibility_status == "INELIGIBLE"


# --- allergy exclusion ---------------------------------------------------------------


def test_allergy_exclusion_pass_when_no_allergy():
    criterion = EligibilityCriterionOut(type="allergy", label="Penicillin", code="7984", display="Penicillin V")
    profile = _profile(allergies=[])
    result = matcher.evaluate(profile, _trial(exclusion_criteria=[criterion]))
    assert result.eligibility_status == "ELIGIBLE"
    assert result.matched_criteria[0].requirement == "exclusion"


def test_allergy_exclusion_fail_when_allergy_present():
    criterion = EligibilityCriterionOut(type="allergy", label="Penicillin", code="7984", display="Penicillin V")
    profile = _profile(allergies=[AllergyOut(resource_id="a1", code="7984", substance="Penicillin V")])
    result = matcher.evaluate(profile, _trial(exclusion_criteria=[criterion]))
    assert result.eligibility_status == "INELIGIBLE"
    assert result.failed_criteria[0].requirement == "exclusion"


# --- multiple criteria / scoring / determinism / explanation ---------------------


def test_multiple_criteria_mixed_pass_fail_unknown():
    inclusion = [
        EligibilityCriterionOut(type="condition", label="Prediabetes", code="15777000", display="Prediabetes"),
        EligibilityCriterionOut(type="observation_threshold", label="HbA1c", code="4548-4", operator=">=", value=7.0),
    ]
    exclusion = [EligibilityCriterionOut(type="allergy", label="Penicillin", code="7984", display="Penicillin V")]
    profile = _profile(
        date_of_birth=_dob_for_age(50),
        conditions=[ConditionOut(resource_id="c1", code="15777000", display="Prediabetes")],
        allergies=[AllergyOut(resource_id="a1", code="7984", substance="Penicillin V")],
    )
    result = matcher.evaluate(profile, _trial(minimum_age=18, inclusion_criteria=inclusion, exclusion_criteria=exclusion))

    assert len(result.matched_criteria) == 2  # age + prediabetes condition
    assert len(result.unknown_criteria) == 1  # HbA1c
    assert len(result.failed_criteria) == 1  # penicillin allergy violates exclusion
    assert result.eligibility_status == "INELIGIBLE"  # FAIL takes precedence over UNKNOWN


def test_fully_eligible_patient_has_full_score():
    profile = _profile(date_of_birth=_dob_for_age(50), gender="female")
    result = matcher.evaluate(profile, _trial(minimum_age=18, sex="female"))
    assert result.match_score == 1.0
    assert result.eligibility_status == "ELIGIBLE"


def test_clearly_ineligible_patient():
    result = matcher.evaluate(_profile(date_of_birth=_dob_for_age(10)), _trial(minimum_age=18))
    assert result.eligibility_status == "INELIGIBLE"
    assert result.match_score == 0.0


def test_match_score_calculation():
    inclusion = [EligibilityCriterionOut(type="condition", label="Prediabetes", code="15777000", display="Prediabetes")]
    profile = _profile(date_of_birth=_dob_for_age(50), conditions=[])
    result = matcher.evaluate(profile, _trial(minimum_age=18, inclusion_criteria=inclusion))
    # age PASS, condition FAIL -> 1 of 2 criteria passed
    assert result.match_score == 0.5


def test_deterministic_repeated_evaluation_is_identical():
    profile = _profile(
        date_of_birth=_dob_for_age(50),
        gender="female",
        conditions=[ConditionOut(resource_id="c1", code="59621000", display="Hypertension")],
    )
    trial = _trial(
        minimum_age=18,
        sex="female",
        inclusion_criteria=[EligibilityCriterionOut(type="condition", label="Hypertension", code="59621000", display="Hypertension")],
    )

    first = matcher.evaluate(profile, trial)
    second = matcher.evaluate(profile, trial)

    assert first.eligibility_status == second.eligibility_status
    assert first.match_score == second.match_score
    first_results = [c.result for c in first.matched_criteria + first.failed_criteria + first.unknown_criteria]
    second_results = [c.result for c in second.matched_criteria + second.failed_criteria + second.unknown_criteria]
    assert first_results == second_results


def test_explanation_reflects_actual_outcome():
    result = matcher.evaluate(_profile(date_of_birth=_dob_for_age(10)), _trial(minimum_age=18))
    assert "INELIGIBLE" in result.explanation
    assert "0 of 1 criteria matched" in result.explanation
    assert "Age >= 18" in result.explanation


def test_no_hallucinated_facts_missing_evidence_stays_unknown():
    # No observations, conditions, medications, or allergies recorded at
    # all — the clinical criterion must come back UNKNOWN, never guessed PASS.
    inclusion = [EligibilityCriterionOut(type="observation_threshold", label="HbA1c", code="4548-4", operator=">=", value=7.0)]
    profile = _profile(date_of_birth=_dob_for_age(40))
    result = matcher.evaluate(profile, _trial(minimum_age=18, inclusion_criteria=inclusion))
    assert result.eligibility_status == "UNKNOWN"
    assert result.unknown_criteria[0].patient_value is None
