import inspect
import shutil
from pathlib import Path

from app.config import get_settings
from app.repositories import trial_repository
from app.services.fhir_ingestion import FHIRIngestionService
from app.services.patient_normalization import PatientNormalizationService
from app.services.trial_ingestion import TrialIngestionService
from app.services.trial_matching import TrialMatchingService


def _setup(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()
    PatientNormalizationService(mongo_db).normalize_all()

    trials_dir = tmp_path / "trials"
    trials_dir.mkdir()
    shutil.copy(fixtures_dir / "trials_matching_fixture.json", trials_dir / "trials_matching_fixture.json")
    TrialIngestionService(mongo_db, trials_path=trials_dir / "trials_matching_fixture.json").run()


def test_match_patient_to_trial_eligible(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    result = TrialMatchingService(mongo_db).match_patient_to_trial("profile-patient-1", "MATCH-TRIAL-ELIGIBLE")

    assert result is not None
    assert result.patient_id == "profile-patient-1"
    assert result.trial_id == "MATCH-TRIAL-ELIGIBLE"
    assert result.eligibility_status == "ELIGIBLE"


def test_match_patient_to_trial_ineligible_due_to_allergy_exclusion(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    result = TrialMatchingService(mongo_db).match_patient_to_trial("profile-patient-1", "MATCH-TRIAL-INELIGIBLE")

    assert result.eligibility_status == "INELIGIBLE"
    assert any(c.category == "allergy" for c in result.failed_criteria)


def test_match_patient_to_trial_unknown_due_to_missing_observation(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    result = TrialMatchingService(mongo_db).match_patient_to_trial("profile-patient-1", "MATCH-TRIAL-UNKNOWN")

    assert result.eligibility_status == "UNKNOWN"
    assert result.unknown_criteria[0].patient_value is None
    assert "HbA1c" in result.unknown_criteria[0].criterion


def test_match_patient_to_trial_unknown_patient_returns_none(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    result = TrialMatchingService(mongo_db).match_patient_to_trial("does-not-exist", "MATCH-TRIAL-ELIGIBLE")

    assert result is None


def test_match_patient_to_trial_unknown_trial_returns_none(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    result = TrialMatchingService(mongo_db).match_patient_to_trial("profile-patient-1", "does-not-exist")

    assert result is None


def test_match_patient_to_trials_excludes_completed_trials_by_default(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    results = TrialMatchingService(mongo_db).match_patient_to_trials("profile-patient-1")

    assert all(r.trial_id != "MATCH-TRIAL-COMPLETED" for r in results)


def test_match_patient_to_trials_status_none_includes_completed_trials(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    results = TrialMatchingService(mongo_db).match_patient_to_trials("profile-patient-1", status=None)

    assert any(r.trial_id == "MATCH-TRIAL-COMPLETED" for r in results)


def test_match_patient_to_trials_ranks_eligible_before_ineligible(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    results = TrialMatchingService(mongo_db).match_patient_to_trials("profile-patient-1")
    statuses = [r.eligibility_status for r in results]

    if "ELIGIBLE" in statuses and "INELIGIBLE" in statuses:
        assert statuses.index("ELIGIBLE") < statuses.index("INELIGIBLE")


def test_match_patient_to_trials_unknown_patient_returns_none(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    results = TrialMatchingService(mongo_db).match_patient_to_trials("does-not-exist")

    assert results is None


def test_patient_isolation_across_trial_matches(mongo_db, tmp_path, fixtures_dir):
    # profile-patient-1 has Hypertension only; profile-patient-2 has Diabetes only.
    _setup(mongo_db, tmp_path, fixtures_dir)
    service = TrialMatchingService(mongo_db)

    result_p1 = service.match_patient_to_trial("profile-patient-1", "MATCH-TRIAL-DIABETES")
    result_p2 = service.match_patient_to_trial("profile-patient-2", "MATCH-TRIAL-DIABETES")

    assert result_p1.eligibility_status == "INELIGIBLE"
    assert result_p2.eligibility_status == "ELIGIBLE"


def test_deterministic_repeated_match(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    service = TrialMatchingService(mongo_db)

    first = service.match_patient_to_trial("profile-patient-1", "MATCH-TRIAL-ELIGIBLE")
    second = service.match_patient_to_trial("profile-patient-1", "MATCH-TRIAL-ELIGIBLE")

    assert first.eligibility_status == second.eligibility_status
    assert first.match_score == second.match_score


# --- scalability hardening: candidate trials must be streamed, not materialized ---


def test_iter_candidate_trials_is_a_lazy_generator_not_a_materialized_list(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    result = trial_repository.iter_candidate_trials(mongo_db, status="recruiting")

    # Must be a lazy generator over the MongoDB cursor, not a pre-built list —
    # calling the function must not itself materialize every candidate trial.
    assert inspect.isgenerator(result)
    assert not isinstance(result, list)

    trials = list(result)
    assert len(trials) == 4  # 5 fixture trials minus the 1 "completed" one
    assert all(t["status"] == "recruiting" for t in trials)


def test_iter_candidate_trials_status_none_returns_every_trial(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    trials = list(trial_repository.iter_candidate_trials(mongo_db, status=None))

    assert len(trials) == 5  # all fixture trials, including the completed one


def test_match_patient_to_trials_handles_all_17_real_dev_trials(mongo_db, tmp_path, fixtures_dir):
    # Uses the actual shipped data/trials/dev_trials.json, not a test fixture,
    # to prove the streamed candidate-trial query still handles the full
    # real trial catalog identically to before this hardening change.
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()
    PatientNormalizationService(mongo_db).normalize_all()

    real_trials_path = Path(get_settings().trials_data_path)
    TrialIngestionService(mongo_db, trials_path=real_trials_path).run()

    all_results = TrialMatchingService(mongo_db).match_patient_to_trials("profile-patient-1", status=None)
    recruiting_results = TrialMatchingService(mongo_db).match_patient_to_trials("profile-patient-1")

    assert len(all_results) == 17
    assert len(recruiting_results) == 16  # DEV-TRIAL-014 is "completed", excluded by default
    assert all(r.trial_id != "DEV-TRIAL-014" for r in recruiting_results)
