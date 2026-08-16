"""Phase 10F: focused contract verification for exactly the 8 endpoints
named in the Phase 10 spec — valid requests, invalid patient, invalid
parameters, empty query, pagination, status codes, schemas, and data
minimization (no raw Mongo _id, no raw FHIR, no raw LLM output). No real
LLM is ever called."""

import shutil

from app.api.routes.patients import get_llm_provider
from app.main import app
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


class _FakeLLMProvider:
    def __init__(self, response: str = "The patient has hypertension [Condition/cond-1]."):
        self._response = response

    def generate(self, prompt: str) -> str:
        return self._response


def _override_llm(fake):
    app.dependency_overrides[get_llm_provider] = lambda: fake


def _clear_llm_override():
    app.dependency_overrides.pop(get_llm_provider, None)


# --- GET /health -----------------------------------------------------------------------------


def test_health_endpoint(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --- GET /api/patients --------------------------------------------------------------------------


def test_patient_listing(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients")
    assert resp.status_code == 200
    body = resp.json()
    assert "pagination" in body
    for item in body["items"]:
        assert "_id" not in item


def test_patient_listing_pagination(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients?page=1&page_size=1")
    body = resp.json()
    assert len(body["items"]) == 1
    assert body["pagination"]["page_size"] == 1


# --- GET /api/patients/{patient_id} ---------------------------------------------------------------


def test_get_patient_valid_and_invalid(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    valid = api_client.get("/api/patients/profile-patient-1")
    assert valid.status_code == 200
    assert "_id" not in valid.json()

    invalid = api_client.get("/api/patients/does-not-exist")
    assert invalid.status_code == 404


# --- GET /api/patients/{patient_id}/resources -----------------------------------------------------


def test_patient_resources_valid_pagination_and_invalid_patient(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/resources?page=1&page_size=3")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) <= 3
    assert body["pagination"]["page_size"] == 3
    for item in body["items"]:
        assert "_id" not in item

    invalid = api_client.get("/api/patients/does-not-exist/resources")
    assert invalid.status_code == 404


# --- GET /api/patients/{patient_id}/evidence -------------------------------------------------------


def test_patient_evidence_valid_pagination_and_no_raw_fhir(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/evidence?page=1&page_size=2")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) <= 2
    assert body["pagination"]["page_size"] == 2
    for item in body["items"]:
        assert "data" not in item
        assert "resourceType" not in item
        assert "_id" not in item

    invalid = api_client.get("/api/patients/does-not-exist/evidence")
    assert invalid.status_code == 404


# --- GET /api/patients/{patient_id}/matches/{trial_id} ----------------------------------------------


def test_patient_trial_match_valid_and_invalid(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    valid = api_client.get("/api/patients/profile-patient-1/matches/MATCH-TRIAL-ELIGIBLE")
    assert valid.status_code == 200
    body = valid.json()
    assert body["trial_id"] == "MATCH-TRIAL-ELIGIBLE"
    assert body["eligibility_status"] in ("ELIGIBLE", "INELIGIBLE", "UNKNOWN")
    assert "_id" not in body
    for evidence in [e for c in body["matched_criteria"] for e in c["evidence"]]:
        assert "data" not in evidence
        assert "_id" not in evidence

    invalid_patient = api_client.get("/api/patients/does-not-exist/matches/MATCH-TRIAL-ELIGIBLE")
    assert invalid_patient.status_code == 404

    invalid_trial = api_client.get("/api/patients/profile-patient-1/matches/DOES-NOT-EXIST")
    assert invalid_trial.status_code == 404


# --- GET /api/patients/{patient_id}/audit -------------------------------------------------------------


def test_patient_audit_valid_pagination_and_no_leakage(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider("The patient has hypertension [Condition/cond-1]. SECRET-TOKEN-xyz"))
    for _ in range(3):
        api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})
    _clear_llm_override()

    resp = api_client.get("/api/patients/profile-patient-1/audit?page=1&page_size=2")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["items"]) == 2
    assert body["pagination"]["total"] == 3
    assert "SECRET-TOKEN-xyz" not in resp.text
    assert "answer_text" not in body["items"][0]

    invalid = api_client.get("/api/patients/does-not-exist/audit")
    assert invalid.status_code == 404


# --- POST /api/patients/{patient_id}/ask ----------------------------------------------------------------


def test_ask_valid_invalid_patient_and_empty_query(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider("The patient has hypertension [Condition/cond-1]."))

    valid = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})
    assert valid.status_code == 200
    assert valid.json()["status"] == "answered"
    assert "data" not in valid.json()

    invalid_patient = api_client.post("/api/patients/does-not-exist/ask", json={"query": "hypertension"})
    assert invalid_patient.status_code == 404

    empty_query = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "   "})
    assert empty_query.status_code == 400

    missing_query_field = api_client.post("/api/patients/profile-patient-1/ask", json={})
    assert missing_query_field.status_code == 422  # invalid parameters — Pydantic request validation

    _clear_llm_override()
