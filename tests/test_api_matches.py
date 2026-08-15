import shutil

from app.services.fhir_ingestion import FHIRIngestionService
from app.services.patient_normalization import PatientNormalizationService
from app.services.trial_ingestion import TrialIngestionService


def _setup(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()
    PatientNormalizationService(mongo_db).normalize_all()

    trials_dir = tmp_path / "trials"
    trials_dir.mkdir()
    shutil.copy(fixtures_dir / "trials_matching_fixture.json", trials_dir / "trials_matching_fixture.json")
    TrialIngestionService(mongo_db, trials_path=trials_dir / "trials_matching_fixture.json").run()


def test_get_patient_matches(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/matches")

    assert resp.status_code == 200
    body = resp.json()
    assert body["patient_id"] == "profile-patient-1"
    assert body["total_trials_evaluated"] > 0
    assert all(m["trial_id"] != "MATCH-TRIAL-COMPLETED" for m in body["matches"])


def test_get_patient_matches_status_all_includes_completed(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/matches?status=all")

    body = resp.json()
    assert any(m["trial_id"] == "MATCH-TRIAL-COMPLETED" for m in body["matches"])


def test_get_patient_matches_unknown_patient_returns_404(api_client):
    resp = api_client.get("/api/patients/does-not-exist/matches")

    assert resp.status_code == 404


def test_get_patient_trial_match_detail(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/matches/MATCH-TRIAL-ELIGIBLE")

    assert resp.status_code == 200
    body = resp.json()
    assert body["eligibility_status"] == "ELIGIBLE"
    assert len(body["matched_criteria"]) > 0
    assert body["matched_criteria"][0]["reason"]
    assert body["explanation"]


def test_get_patient_trial_match_unknown_evidence_returns_unknown(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/matches/MATCH-TRIAL-UNKNOWN")

    assert resp.status_code == 200
    body = resp.json()
    assert body["eligibility_status"] == "UNKNOWN"
    assert body["unknown_criteria"][0]["patient_value"] is None


def test_get_patient_trial_match_unknown_patient_404(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/does-not-exist/matches/MATCH-TRIAL-ELIGIBLE")

    assert resp.status_code == 404


def test_get_patient_trial_match_unknown_trial_404(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/matches/does-not-exist")

    assert resp.status_code == 404


def test_matches_do_not_leak_across_patients(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp1 = api_client.get("/api/patients/profile-patient-1/matches/MATCH-TRIAL-DIABETES")
    resp2 = api_client.get("/api/patients/profile-patient-2/matches/MATCH-TRIAL-DIABETES")

    assert resp1.json()["eligibility_status"] == "INELIGIBLE"
    assert resp2.json()["eligibility_status"] == "ELIGIBLE"
