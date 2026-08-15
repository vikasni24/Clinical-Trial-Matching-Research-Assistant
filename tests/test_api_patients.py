import shutil

from app.services.fhir_ingestion import FHIRIngestionService


def _ingest_both(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "patient_standalone.json", tmp_path / "patient_standalone.json")
    shutil.copy(fixtures_dir / "bundle_sample.json", tmp_path / "bundle_sample.json")
    return FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()


def test_list_patients_empty(api_client):
    resp = api_client.get("/api/patients")

    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert body["pagination"]["total"] == 0


def test_list_patients_after_ingestion(api_client, mongo_db, tmp_path, fixtures_dir):
    _ingest_both(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients")

    assert resp.status_code == 200
    body = resp.json()
    assert body["pagination"]["total"] == 2
    assert len(body["items"]) == 2


def test_get_patient_by_id(api_client, mongo_db, tmp_path, fixtures_dir):
    _ingest_both(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/patient-001")

    assert resp.status_code == 200
    assert resp.json()["patient_id"] == "patient-001"
    assert resp.json()["data"]["resourceType"] == "Patient"


def test_get_patient_not_found(api_client):
    resp = api_client.get("/api/patients/does-not-exist")

    assert resp.status_code == 404


def test_pagination_page_size(api_client, mongo_db, tmp_path, fixtures_dir):
    _ingest_both(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients?page=1&page_size=1")

    body = resp.json()
    assert len(body["items"]) == 1
    assert body["pagination"]["page_size"] == 1
    assert body["pagination"]["total_pages"] == 2


def test_pagination_second_page(api_client, mongo_db, tmp_path, fixtures_dir):
    _ingest_both(mongo_db, tmp_path, fixtures_dir)

    page1 = api_client.get("/api/patients?page=1&page_size=1").json()
    page2 = api_client.get("/api/patients?page=2&page_size=1").json()

    assert page1["items"][0]["patient_id"] != page2["items"][0]["patient_id"]


def test_page_size_is_capped(api_client):
    resp = api_client.get("/api/patients?page=1&page_size=99999")

    assert resp.status_code == 200
    assert resp.json()["pagination"]["page_size"] <= 100
