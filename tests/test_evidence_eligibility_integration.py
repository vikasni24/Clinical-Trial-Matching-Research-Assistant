"""Phase 4D: proves EvidenceService is correctly wired into the
deterministic eligibility flow (TrialMatchingService -> EligibilityMatcher
-> EvidenceService -> EvidenceRepository -> MongoDB) WITHOUT changing any
Phase 3 PASS/FAIL/UNKNOWN decision, score, ranking, or isolation behavior —
those are exhaustively covered already by test_eligibility_matcher.py and
test_trial_matching_service.py, which must all still pass unmodified.

This file focuses specifically on what's new: evidence attached to
criterion evaluations, and the query behavior of that attachment."""

import inspect
import shutil

from app.models.clinical_trial import ClinicalTrialOut, EligibilityCriterionOut, TrialEligibilityOut
from app.models.patient_profile import ContactInfoOut, DemographicsOut, ObservationOut, PatientProfileOut
from app.repositories import patient_profile_repository
from app.services.eligibility_matcher import EligibilityMatcher
from app.services.evidence_service import EvidenceService
from app.services.fhir_ingestion import FHIRIngestionService
from app.services.patient_normalization import PatientNormalizationService
from app.services.trial_ingestion import TrialIngestionService
from app.services.trial_matching import TrialMatchingService


def _load_profile(mongo_db, patient_id: str) -> PatientProfileOut:
    return PatientProfileOut(**patient_profile_repository.get_patient_profile(mongo_db, patient_id))


def _setup(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()
    PatientNormalizationService(mongo_db).normalize_all()

    trials_dir = tmp_path / "trials"
    trials_dir.mkdir()
    shutil.copy(fixtures_dir / "trials_matching_fixture.json", trials_dir / "trials_matching_fixture.json")
    TrialIngestionService(mongo_db, trials_path=trials_dir / "trials_matching_fixture.json").run()


# --- observation evidence drives PASS/FAIL/UNKNOWN, with evidence attached ---


def test_matching_observation_produces_pass_with_evidence(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    trial = ClinicalTrialOut(
        trial_id="BP-TRIAL",
        status="recruiting",
        eligibility=TrialEligibilityOut(
            inclusion_criteria=[
                EligibilityCriterionOut(
                    type="observation_threshold", label="Systolic BP", code="8480-6", operator=">=", value=120, unit="mm[Hg]"
                )
            ]
        ),
    )
    result = TrialMatchingService(mongo_db)._matcher.evaluate(_load_profile(mongo_db, "profile-patient-1"), trial)

    assert result.eligibility_status == "ELIGIBLE"
    criterion = result.matched_criteria[0]
    assert criterion.result == "PASS"
    assert criterion.patient_value == 125
    assert len(criterion.evidence) == 1
    assert criterion.evidence[0].resource_id == "obs-1"
    assert criterion.evidence[0].resource_type == "Observation"
    assert criterion.evidence[0].value == 125


def test_non_matching_observation_produces_fail_with_evidence(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    trial = ClinicalTrialOut(
        trial_id="BP-TRIAL-HIGH",
        status="recruiting",
        eligibility=TrialEligibilityOut(
            inclusion_criteria=[
                EligibilityCriterionOut(
                    type="observation_threshold", label="Systolic BP", code="8480-6", operator=">=", value=200, unit="mm[Hg]"
                )
            ]
        ),
    )
    service = TrialMatchingService(mongo_db)
    profile = _load_profile(mongo_db, "profile-patient-1")
    result = service._matcher.evaluate(profile, trial)

    assert result.eligibility_status == "INELIGIBLE"
    criterion = result.failed_criteria[0]
    assert criterion.result == "FAIL"
    assert criterion.patient_value == 125  # the real recorded value, not fabricated
    assert len(criterion.evidence) == 1
    assert criterion.evidence[0].resource_id == "obs-1"
    assert criterion.evidence[0].value == 125


def test_missing_observation_produces_unknown_with_no_evidence(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    result = TrialMatchingService(mongo_db).match_patient_to_trial("profile-patient-1", "MATCH-TRIAL-UNKNOWN")

    assert result.eligibility_status == "UNKNOWN"
    criterion = result.unknown_criteria[0]
    assert criterion.result == "UNKNOWN"
    assert criterion.patient_value is None
    assert criterion.evidence == []  # nothing was found — no fabricated evidence


# --- condition / medication / allergy: existing behavior preserved, evidence attached ---


def test_condition_evidence_preserves_existing_behavior(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    result = TrialMatchingService(mongo_db).match_patient_to_trial("profile-patient-1", "MATCH-TRIAL-ELIGIBLE")

    assert result.eligibility_status == "ELIGIBLE"
    condition_eval = next(c for c in result.matched_criteria if c.category == "condition")
    assert condition_eval.patient_value is True
    assert len(condition_eval.evidence) == 1
    assert condition_eval.evidence[0].resource_type == "Condition"
    assert condition_eval.evidence[0].resource_id == "cond-1"
    assert condition_eval.evidence[0].display == "Hypertension"


def test_medication_evidence_preserves_existing_behavior(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    trial = ClinicalTrialOut(
        trial_id="MED-TRIAL",
        status="recruiting",
        eligibility=TrialEligibilityOut(
            inclusion_criteria=[
                EligibilityCriterionOut(type="medication", label="Metformin", code="860975", display="Metformin")
            ]
        ),
    )
    profile = _load_profile(mongo_db, "profile-patient-1")
    result = TrialMatchingService(mongo_db)._matcher.evaluate(profile, trial)

    assert result.eligibility_status == "ELIGIBLE"
    med_eval = result.matched_criteria[0]
    assert med_eval.patient_value is True
    assert len(med_eval.evidence) == 1
    assert med_eval.evidence[0].resource_type == "MedicationRequest"
    assert med_eval.evidence[0].resource_id == "med-1"


def test_allergy_evidence_preserves_existing_behavior(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    result = TrialMatchingService(mongo_db).match_patient_to_trial("profile-patient-1", "MATCH-TRIAL-INELIGIBLE")

    assert result.eligibility_status == "INELIGIBLE"
    allergy_eval = next(c for c in result.failed_criteria if c.category == "allergy")
    assert allergy_eval.patient_value is True
    assert len(allergy_eval.evidence) == 1
    assert allergy_eval.evidence[0].resource_type == "AllergyIntolerance"
    assert allergy_eval.evidence[0].resource_id == "allergy-1"


# --- age / sex: unchanged behavior, now with evidence pointing at the Patient resource ---


def test_age_behavior_unchanged_with_evidence(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    result = TrialMatchingService(mongo_db).match_patient_to_trial("profile-patient-1", "MATCH-TRIAL-ELIGIBLE")

    age_eval = next(c for c in result.matched_criteria if c.category == "demographics")
    assert age_eval.result == "PASS"
    assert len(age_eval.evidence) == 1
    assert age_eval.evidence[0].resource_type == "Patient"
    assert age_eval.evidence[0].resource_id == "profile-patient-1"


def test_sex_behavior_unchanged_with_evidence(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    trial = ClinicalTrialOut(
        trial_id="SEX-TRIAL",
        status="recruiting",
        eligibility=TrialEligibilityOut(sex="female"),
    )
    profile = _load_profile(mongo_db, "profile-patient-1")
    result = TrialMatchingService(mongo_db)._matcher.evaluate(profile, trial)

    assert result.eligibility_status == "ELIGIBLE"
    sex_eval = result.matched_criteria[0]
    assert sex_eval.patient_value == "female"
    assert len(sex_eval.evidence) == 1
    assert sex_eval.evidence[0].resource_type == "Patient"
    assert sex_eval.evidence[0].resource_id == "profile-patient-1"


def test_age_unknown_when_dob_missing_has_no_evidence(mongo_db):
    profile = PatientProfileOut(
        patient_id="no-dob-patient",
        demographics=DemographicsOut(patient_id="no-dob-patient"),
        contact=ContactInfoOut(),
    )
    trial = ClinicalTrialOut(
        trial_id="AGE-TRIAL", status="recruiting", eligibility=TrialEligibilityOut(minimum_age=18)
    )
    matcher = EligibilityMatcher(evidence_service=EvidenceService(mongo_db))

    result = matcher.evaluate(profile, trial)

    assert result.eligibility_status == "UNKNOWN"
    assert result.unknown_criteria[0].evidence == []


# --- multiple observations: existing most-recent rule preserved, evidence matches it ---


def test_multiple_observations_preserve_most_recent_rule_and_matching_evidence(mongo_db):
    patient_id = "multi-obs-patient"
    mongo_db["fhir_resources"].insert_many(
        [
            {
                "patient_id": patient_id,
                "resource_type": "Observation",
                "resource_id": "obs-old",
                "data": {
                    "resourceType": "Observation",
                    "id": "obs-old",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "2339-0", "display": "Glucose"}], "text": "Glucose"},
                    "effectiveDateTime": "2020-01-01",
                    "valueQuantity": {"value": 80, "unit": "mg/dL"},
                },
            },
            {
                "patient_id": patient_id,
                "resource_type": "Observation",
                "resource_id": "obs-new",
                "data": {
                    "resourceType": "Observation",
                    "id": "obs-new",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "2339-0", "display": "Glucose"}], "text": "Glucose"},
                    "effectiveDateTime": "2023-06-01",
                    "valueQuantity": {"value": 150, "unit": "mg/dL"},
                },
            },
        ]
    )

    profile = PatientProfileOut(
        patient_id=patient_id,
        demographics=DemographicsOut(patient_id=patient_id, date_of_birth="1980-01-01", gender="female"),
        contact=ContactInfoOut(),
        observations=[
            ObservationOut(resource_id="obs-old", code="2339-0", name="Glucose", value=80, value_type="Quantity", unit="mg/dL", effective_date="2020-01-01"),
            ObservationOut(resource_id="obs-new", code="2339-0", name="Glucose", value=150, value_type="Quantity", unit="mg/dL", effective_date="2023-06-01"),
        ],
    )
    trial = ClinicalTrialOut(
        trial_id="GLUCOSE-TRIAL",
        status="recruiting",
        eligibility=TrialEligibilityOut(
            inclusion_criteria=[
                EligibilityCriterionOut(type="observation_threshold", label="Glucose", code="2339-0", operator=">=", value=100, unit="mg/dL")
            ]
        ),
    )

    matcher = EligibilityMatcher(evidence_service=EvidenceService(mongo_db))
    result = matcher.evaluate(profile, trial)

    assert result.eligibility_status == "ELIGIBLE"
    glucose_eval = result.matched_criteria[0]
    assert glucose_eval.patient_value == 150  # most-recent value — unchanged Phase 3 rule
    assert len(glucose_eval.evidence) == 1
    assert glucose_eval.evidence[0].resource_id == "obs-new"  # evidence traces to that same resource


# --- patient isolation --------------------------------------------------------


def test_patient_isolation_in_evidence_attachment(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    service = TrialMatchingService(mongo_db)

    result_p1 = service.match_patient_to_trial("profile-patient-1", "MATCH-TRIAL-DIABETES")
    result_p2 = service.match_patient_to_trial("profile-patient-2", "MATCH-TRIAL-DIABETES")

    # Patient 1 (no Diabetes) must never receive patient 2's Condition evidence.
    p1_condition_eval = next(c for c in result_p1.failed_criteria if c.category == "condition")
    assert p1_condition_eval.evidence == []
    assert result_p1.eligibility_status == "INELIGIBLE"

    # Patient 2 (has Diabetes) gets evidence for their own cond-2, never cond-1.
    p2_condition_eval = next(c for c in result_p2.matched_criteria if c.category == "condition")
    assert p2_condition_eval.evidence[0].resource_id == "cond-2"
    assert p2_condition_eval.evidence[0].patient_id == "profile-patient-2"
    assert result_p2.eligibility_status == "ELIGIBLE"

    for evaluation in result_p1.matched_criteria + result_p1.failed_criteria + result_p1.unknown_criteria:
        for evidence in evaluation.evidence:
            assert evidence.patient_id == "profile-patient-1"
    for evaluation in result_p2.matched_criteria + result_p2.failed_criteria + result_p2.unknown_criteria:
        for evidence in evaluation.evidence:
            assert evidence.patient_id == "profile-patient-2"


def test_patient_a_cannot_receive_patient_b_evidence_across_all_categories(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    trial = ClinicalTrialOut(
        trial_id="ALL-CATEGORIES-TRIAL",
        status="recruiting",
        eligibility=TrialEligibilityOut(
            inclusion_criteria=[
                EligibilityCriterionOut(type="condition", label="Hypertension", code="38341003", display="Hypertension"),
                EligibilityCriterionOut(type="medication", label="Metformin", code="860975", display="Metformin"),
                EligibilityCriterionOut(
                    type="observation_threshold", label="Systolic BP", code="8480-6", operator=">=", value=100, unit="mm[Hg]"
                ),
            ],
            exclusion_criteria=[
                EligibilityCriterionOut(type="allergy", label="Peanut", code="91935009", display="Peanut"),
            ],
        ),
    )
    # profile-patient-2 has none of patient-1's Condition/Medication/Observation/Allergy records.
    profile_2 = _load_profile(mongo_db, "profile-patient-2")
    matcher = EligibilityMatcher(evidence_service=EvidenceService(mongo_db))
    result = matcher.evaluate(profile_2, trial)

    all_evaluations = result.matched_criteria + result.failed_criteria + result.unknown_criteria
    assert len(all_evaluations) == 4
    for evaluation in all_evaluations:
        # patient-2 has none of these resources — every evidence list must be empty,
        # never patient-1's cond-1/med-1/obs-1/allergy-1.
        assert evaluation.evidence == []
        for evidence in evaluation.evidence:
            assert evidence.patient_id != "profile-patient-1"


# --- determinism ---------------------------------------------------------------


def test_deterministic_repeated_result_including_evidence(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    service = TrialMatchingService(mongo_db)

    first = service.match_patient_to_trial("profile-patient-1", "MATCH-TRIAL-ELIGIBLE")
    second = service.match_patient_to_trial("profile-patient-1", "MATCH-TRIAL-ELIGIBLE")

    first_dump = first.model_dump(exclude={"evaluated_at"})
    second_dump = second.model_dump(exclude={"evaluated_at"})

    assert first_dump == second_dump


# --- no raw FHIR document embedded / no Phase 5 functionality -----------------


def test_no_raw_fhir_document_embedded_in_match_result(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    result = TrialMatchingService(mongo_db).match_patient_to_trial("profile-patient-1", "MATCH-TRIAL-ELIGIBLE")

    dumped = result.model_dump()
    # Every evidence item must only ever carry the flat Evidence fields —
    # never a nested raw FHIR document.
    for evaluation_list in (dumped["matched_criteria"], dumped["failed_criteria"], dumped["unknown_criteria"]):
        for evaluation in evaluation_list:
            assert "data" not in evaluation
            for evidence in evaluation["evidence"]:
                assert set(evidence.keys()).issubset(
                    {
                        "patient_id",
                        "resource_type",
                        "resource_id",
                        "source_collection",
                        "source_reference",
                        "code",
                        "coding_system",
                        "display",
                        "value",
                        "unit",
                        "effective_date",
                        "status",
                    }
                )


def test_no_phase5_functionality_involved():
    # Module docstrings legitimately contain negation prose like "no
    # embeddings" as an explicit disclaimer — so this checks for actual
    # usage indicators (imports/library names), not the word alone.
    import app.services.eligibility_matcher as matcher_module
    import app.services.trial_matching as matching_module

    for module in (matcher_module, matching_module):
        source = inspect.getsource(module).lower()
        for forbidden in ("chromadb", "pinecone", "faiss", "openai", "anthropic", "langchain", "import embed", "vectorstore", "vector_db", "vector store"):
            assert forbidden not in source


# --- query behavior: evidence attachment stays targeted, never unbounded -------


def test_evidence_attachment_queries_stay_patient_scoped(mongo_db, tmp_path, fixtures_dir, monkeypatch):
    _setup(mongo_db, tmp_path, fixtures_dir)
    collection = mongo_db["fhir_resources"]
    original_find_one = collection.find_one
    captured = []

    def spying_find_one(filter=None, *args, **kwargs):
        captured.append(filter or {})
        return original_find_one(filter, *args, **kwargs)

    monkeypatch.setattr(collection, "find_one", spying_find_one)

    TrialMatchingService(mongo_db).match_patient_to_trial("profile-patient-1", "MATCH-TRIAL-ELIGIBLE")

    assert captured  # evidence lookups actually happened
    for query in captured:
        assert query != {}
        assert query.get("patient_id") == "profile-patient-1"
        assert "resource_type" in query
        assert "resource_id" in query
