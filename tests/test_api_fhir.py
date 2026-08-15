import shutil

from app.services.fhir_ingestion import FHIRIngestionService


def _ingest_bundle(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "bundle_sample.json", tmp_path / "bundle_sample.json")
    return FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()


def test_get_patient_resources(api_client, mongo_db, tmp_path, fixtures_dir):
    _ingest_bundle(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/patient-002/resources")

    assert resp.status_code == 200
    body = resp.json()
    assert body["pagination"]["total"] == 3  # Patient + Observation + Condition


def test_get_patient_resources_filtered_by_type(api_client, mongo_db, tmp_path, fixtures_dir):
    _ingest_bundle(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/patient-002/resources?resource_type=Observation")

    assert resp.status_code == 200
    body = resp.json()
    assert body["pagination"]["total"] == 1
    assert body["items"][0]["resource_type"] == "Observation"
    assert body["items"][0]["resource_id"] == "obs-100"


def test_get_patient_resources_for_unknown_patient_returns_404(api_client):
    resp = api_client.get("/api/patients/unknown/resources")

    assert resp.status_code == 404


def test_patient_resources_do_not_leak_across_patients(api_client, mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "patient_standalone.json", tmp_path / "patient_standalone.json")
    _ingest_bundle(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/patient-001/resources")

    body = resp.json()
    assert body["pagination"]["total"] == 1
    assert body["items"][0]["resource_id"] == "patient-001"


def test_get_single_fhir_resource(api_client, mongo_db, tmp_path, fixtures_dir):
    _ingest_bundle(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/fhir/Observation/obs-100")

    assert resp.status_code == 200
    body = resp.json()
    assert body["resource_id"] == "obs-100"
    assert body["data"]["resourceType"] == "Observation"
    assert body["data"]["code"]["text"] == "Heart rate"


def test_get_single_fhir_resource_not_found(api_client):
    resp = api_client.get("/api/fhir/Observation/does-not-exist")

    assert resp.status_code == 404
