"""Phase 9 Part E: one continuous walk through every major existing API
workflow against the same ingested dataset, verifying status codes,
response schemas, patient isolation, empty/unsupported states, pagination,
and data-minimization (no raw FHIR, no MongoDB _id, no raw LLM output)
end-to-end. Individual endpoints already have dedicated unit-level test
files (test_api_patients.py, test_api_evidence.py, test_api_trials.py,
test_api_matches.py, test_api_ask.py, test_api_audit.py); this file proves
the whole surface holds together as one coherent flow. No real LLM is ever
called."""

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


def test_full_api_workflow_walk(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    # 1. Health endpoint.
    health = api_client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    # 2. Patient listing.
    patients = api_client.get("/api/patients")
    assert patients.status_code == 200
    patients_body = patients.json()
    assert "pagination" in patients_body
    assert any(p["patient_id"] == "profile-patient-1" for p in patients_body["items"])
    for p in patients_body["items"]:
        assert "_id" not in p

    # 3. Patient resources (paginated, no raw MongoDB _id).
    resources = api_client.get("/api/patients/profile-patient-1/resources?page=1&page_size=5")
    assert resources.status_code == 200
    resources_body = resources.json()
    assert resources_body["pagination"]["page_size"] == 5
    assert len(resources_body["items"]) <= 5
    for item in resources_body["items"]:
        assert "_id" not in item

    # 4. Patient evidence (no raw FHIR).
    evidence = api_client.get("/api/patients/profile-patient-1/evidence")
    assert evidence.status_code == 200
    evidence_body = evidence.json()
    assert evidence_body["pagination"]["total"] > 0
    for item in evidence_body["items"]:
        assert "data" not in item
        assert "resourceType" not in item
        assert "_id" not in item
        assert item["patient_id"] == "profile-patient-1"

    # 5. Trial listing.
    trials = api_client.get("/api/trials")
    assert trials.status_code == 200
    trials_body = trials.json()
    assert "pagination" in trials_body
    assert "disclaimer" in trials_body
    assert len(trials_body["items"]) > 0

    # 6. Trial matching (all candidate trials for the patient).
    matches = api_client.get("/api/patients/profile-patient-1/matches?status=all")
    assert matches.status_code == 200
    matches_body = matches.json()
    assert matches_body["patient_id"] == "profile-patient-1"
    assert matches_body["total_trials_evaluated"] == len(matches_body["matches"])
    for match in matches_body["matches"]:
        assert match["eligibility_status"] in ("ELIGIBLE", "INELIGIBLE", "UNKNOWN")

    # 7. Patient match for one specific trial.
    trial_id = matches_body["matches"][0]["trial_id"]
    single_match = api_client.get(f"/api/patients/profile-patient-1/matches/{trial_id}")
    assert single_match.status_code == 200
    assert single_match.json()["trial_id"] == trial_id

    # 8. Patient /ask endpoint.
    _override_llm(_FakeLLMProvider("The patient has hypertension [Condition/cond-1]."))
    ask = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})
    assert ask.status_code == 200
    ask_body = ask.json()
    assert ask_body["status"] == "answered"
    assert ask_body["evidence"][0]["resource_id"] == "cond-1"
    assert "data" not in ask_body
    for item in ask_body["evidence"]:
        assert "data" not in item and "_id" not in item

    # 9. Patient audit history — reflects the /ask call above.
    audit = api_client.get("/api/patients/profile-patient-1/audit")
    _clear_llm_override()
    assert audit.status_code == 200
    audit_body = audit.json()
    assert audit_body["pagination"]["total"] == 1
    audit_item = audit_body["items"][0]
    assert audit_item["answer_status"] == "answered"
    assert set(audit_item.keys()) == {
        "audit_id", "patient_id", "query", "retrieval_status", "answer_status", "evidence_references", "created_at",
    }
    assert "answer_text" not in audit_item
    assert "prompt" not in audit_item


def test_unknown_patient_returns_404_across_every_patient_scoped_endpoint(api_client):
    unknown = "does-not-exist"

    assert api_client.get(f"/api/patients/{unknown}").status_code == 404
    assert api_client.get(f"/api/patients/{unknown}/resources").status_code == 404
    assert api_client.get(f"/api/patients/{unknown}/evidence").status_code == 404
    assert api_client.get(f"/api/patients/{unknown}/profile").status_code == 404
    assert api_client.get(f"/api/patients/{unknown}/matches").status_code == 404
    assert api_client.post(f"/api/patients/{unknown}/ask", json={"query": "hypertension"}).status_code == 404
    assert api_client.get(f"/api/patients/{unknown}/audit").status_code == 404


def test_empty_and_unsupported_states_are_reported_correctly(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider())

    unsupported = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "???"})
    no_evidence = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hemoglobin a1c"})
    empty_query = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "   "})
    empty_evidence_page = api_client.get(
        "/api/patients/profile-patient-1/evidence?resource_type=DoesNotExist"
    )

    _clear_llm_override()
    assert unsupported.json()["status"] == "unsupported"
    assert no_evidence.json()["status"] == "insufficient_evidence"
    assert empty_query.status_code == 400
    assert empty_evidence_page.json()["items"] == []
    assert empty_evidence_page.json()["pagination"]["total"] == 0
