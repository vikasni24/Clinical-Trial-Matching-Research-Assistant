"""Phase 7F: GET /api/patients/{patient_id}/audit — patient-scoped, paginated,
newest-first audit history. A real LLM is never called from these tests;
audit records are created via POST /ask with a fake provider override."""

import shutil

from app.api.routes.patients import get_llm_provider
from app.main import app
from app.services.fhir_ingestion import FHIRIngestionService
from app.services.patient_normalization import PatientNormalizationService


def _setup(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()
    PatientNormalizationService(mongo_db).normalize_all()


class _FakeLLMProvider:
    def __init__(self, response: str = "The patient has hypertension [Condition/cond-1]."):
        self._response = response

    def generate(self, prompt: str) -> str:
        return self._response


def _override_llm(fake):
    app.dependency_overrides[get_llm_provider] = lambda: fake


def _clear_llm_override():
    app.dependency_overrides.pop(get_llm_provider, None)


def _ask(api_client, patient_id, query):
    return api_client.post(f"/api/patients/{patient_id}/ask", json={"query": query})


# --- patient must exist -------------------------------------------------------------------


def test_audit_endpoint_unknown_patient_returns_404(api_client):
    resp = api_client.get("/api/patients/does-not-exist/audit")

    assert resp.status_code == 404


# --- patient-scoped / newest-first / pagination -------------------------------------------


def test_audit_history_reflects_asked_questions(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider("The patient has hypertension [Condition/cond-1]."))

    _ask(api_client, "profile-patient-1", "hypertension")
    resp = api_client.get("/api/patients/profile-patient-1/audit")

    _clear_llm_override()
    assert resp.status_code == 200
    body = resp.json()
    assert body["pagination"]["total"] == 1
    item = body["items"][0]
    assert item["patient_id"] == "profile-patient-1"
    assert item["query"] == "hypertension"
    assert item["answer_status"] == "answered"
    assert item["retrieval_status"] == "evidence_found"
    assert item["evidence_references"] == [{"resource_type": "Condition", "resource_id": "cond-1"}]
    assert "audit_id" in item and item["audit_id"]
    assert "created_at" in item


def test_audit_history_newest_first(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider("The patient has hypertension [Condition/cond-1]."))

    _ask(api_client, "profile-patient-1", "hypertension")
    _ask(api_client, "profile-patient-1", "???")
    resp = api_client.get("/api/patients/profile-patient-1/audit")

    _clear_llm_override()
    items = resp.json()["items"]
    assert len(items) == 2
    # The most recently asked question (the unsupported one) comes first.
    assert items[0]["query"] == "???"
    assert items[1]["query"] == "hypertension"


def test_audit_history_pagination(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider())

    for _ in range(3):
        _ask(api_client, "profile-patient-1", "hypertension")
    resp = api_client.get("/api/patients/profile-patient-1/audit?page=1&page_size=2")

    _clear_llm_override()
    body = resp.json()
    assert len(body["items"]) == 2
    assert body["pagination"]["total"] == 3
    assert body["pagination"]["page_size"] == 2
    assert body["pagination"]["total_pages"] == 2


def test_audit_history_is_patient_scoped(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider("The patient has diabetes [Condition/cond-2]."))

    _ask(api_client, "profile-patient-2", "diabetes")
    resp_1 = api_client.get("/api/patients/profile-patient-1/audit")
    resp_2 = api_client.get("/api/patients/profile-patient-2/audit")

    _clear_llm_override()
    assert resp_1.json()["pagination"]["total"] == 0
    assert resp_2.json()["pagination"]["total"] == 1
    assert resp_2.json()["items"][0]["patient_id"] == "profile-patient-2"


# --- no raw FHIR / no raw LLM output / no secrets / no hidden reasoning -------------------


def test_audit_response_contains_no_raw_fhir(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider("The patient has hypertension [Condition/cond-1]."))

    _ask(api_client, "profile-patient-1", "hypertension")
    resp = api_client.get("/api/patients/profile-patient-1/audit")

    _clear_llm_override()
    body_text = resp.text
    for marker in ("resourceType", "clinicalStatus", "valueQuantity", '"_id"', "fhir_resources"):
        assert marker not in body_text
    for item in resp.json()["items"]:
        assert "data" not in item
        for evidence in item["evidence_references"]:
            assert set(evidence.keys()) == {"resource_type", "resource_id"}


def test_audit_response_contains_no_raw_llm_output_prompt_or_secrets(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    raw_llm_text = "The patient has hypertension [Condition/cond-1]. CONFIDENTIAL-LLM-MARKER-XYZ"
    _override_llm(_FakeLLMProvider(raw_llm_text))

    _ask(api_client, "profile-patient-1", "hypertension")
    resp = api_client.get("/api/patients/profile-patient-1/audit")

    _clear_llm_override()
    item = resp.json()["items"][0]
    allowed_keys = {"audit_id", "patient_id", "query", "retrieval_status", "answer_status", "evidence_references", "created_at"}
    assert set(item.keys()) == allowed_keys
    assert "CONFIDENTIAL-LLM-MARKER-XYZ" not in resp.text
    assert "answer_text" not in item
    for marker in ("api_key", "Bearer", "x-api-key", "chain_of_thought", "confidence_score", "hallucination"):
        assert marker not in resp.text
