import shutil

from app.services.fhir_ingestion import FHIRIngestionService
from app.services.patient_normalization import PatientNormalizationService


def _ingest_and_normalize(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()
    PatientNormalizationService(mongo_db).normalize_all()


def test_get_patient_profile(api_client, mongo_db, tmp_path, fixtures_dir):
    _ingest_and_normalize(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/profile")

    assert resp.status_code == 200
    body = resp.json()
    assert body["patient_id"] == "profile-patient-1"
    assert body["demographics"]["full_name"] == "Jane Q Doe"
    assert body["contact"]["city"] == "Boston"
    assert len(body["conditions"]) == 1
    assert body["conditions"][0]["display"] == "Hypertension"
    assert len(body["allergies"]) == 1


def test_get_patient_profile_does_not_leak_other_patient_data(api_client, mongo_db, tmp_path, fixtures_dir):
    _ingest_and_normalize(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-2/profile")

    assert resp.status_code == 200
    body = resp.json()
    assert body["patient_id"] == "profile-patient-2"
    assert [c["resource_id"] for c in body["conditions"]] == ["cond-2"]


def test_get_patient_profile_unknown_patient_returns_404(api_client):
    resp = api_client.get("/api/patients/does-not-exist/profile")

    assert resp.status_code == 404


def test_get_patient_profile_not_yet_normalized_returns_404(api_client, mongo_db, tmp_path, fixtures_dir):
    # Patient exists (ingested) but normalization hasn't run yet for them.
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()

    resp = api_client.get("/api/patients/profile-patient-1/profile")

    assert resp.status_code == 404


def test_existing_patient_endpoints_unaffected_by_profile_route(api_client, mongo_db, tmp_path, fixtures_dir):
    # Phase 1 behavior must be untouched by the new /profile route.
    _ingest_and_normalize(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1")
    assert resp.status_code == 200
    assert resp.json()["patient_id"] == "profile-patient-1"

    resp = api_client.get("/api/patients/profile-patient-1/resources")
    assert resp.status_code == 200
    assert resp.json()["pagination"]["total"] > 0
