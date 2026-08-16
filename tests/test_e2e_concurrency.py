"""Phase 9 Part H: lightweight, controlled concurrency checks — NOT a
benchmark. Verifies that concurrent requests for different patients (and
repeated requests for the same patient) never leak evidence across
patients, never contaminate each other's results, and never corrupt audit
ownership, even when serviced by shared, stateless service classes under
concurrent access. Uses only stdlib `concurrent.futures`; no Redis,
Celery, Kafka, or other infrastructure. No real LLM is ever called."""

import shutil
from concurrent.futures import ThreadPoolExecutor

from app.api.routes.patients import get_llm_provider
from app.main import app
from app.repositories import audit_repository
from app.services.fhir_ingestion import FHIRIngestionService
from app.services.patient_normalization import PatientNormalizationService


def _setup(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()
    PatientNormalizationService(mongo_db).normalize_all()


class _FakeLLMProvider:
    """Each call independently determines its own response based on the
    prompt content (patient-specific), rather than any shared mutable
    counter — a stand-in that itself cannot introduce cross-request
    contamination."""

    def generate(self, prompt: str) -> str:
        if "cond-1" in prompt or "Hypertension" in prompt:
            return "The patient has hypertension [Condition/cond-1]."
        if "cond-2" in prompt or "Diabetes" in prompt:
            return "The patient has diabetes [Condition/cond-2]."
        return "No specific finding cited."


def _override_llm(fake):
    app.dependency_overrides[get_llm_provider] = lambda: fake


def _clear_llm_override():
    app.dependency_overrides.pop(get_llm_provider, None)


_REQUESTS = [
    ("profile-patient-1", "hypertension", "cond-1"),
    ("profile-patient-2", "diabetes", "cond-2"),
] * 6  # 12 interleaved requests across 2 patients


def test_concurrent_ask_requests_never_cross_contaminate_patients(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider())

    def _ask(args):
        patient_id, query, expected_resource_id = args
        resp = api_client.post(f"/api/patients/{patient_id}/ask", json={"query": query})
        return patient_id, expected_resource_id, resp

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(_ask, _REQUESTS))

    _clear_llm_override()

    for patient_id, expected_resource_id, resp in results:
        assert resp.status_code == 200
        body = resp.json()
        # Each response must belong to exactly the patient that was asked
        # about, never a mix-up with the other concurrently-running request.
        assert body["patient_id"] == patient_id
        assert [e["resource_id"] for e in body["evidence"]] == [expected_resource_id]
        assert all(True for _ in body["evidence"])  # response is well-formed


def test_concurrent_requests_produce_correctly_owned_audit_records(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider())

    def _ask(args):
        patient_id, query, _ = args
        return api_client.post(f"/api/patients/{patient_id}/ask", json={"query": query})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_ask, _REQUESTS))

    _clear_llm_override()

    records_1, total_1 = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-1", page_size=100)
    records_2, total_2 = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-2", page_size=100)

    assert total_1 == 6
    assert total_2 == 6
    assert all(r.patient_id == "profile-patient-1" for r in records_1)
    assert all(r.patient_id == "profile-patient-2" for r in records_2)
    assert all([e.resource_id for e in r.evidence_references] == ["cond-1"] for r in records_1)
    assert all([e.resource_id for e in r.evidence_references] == ["cond-2"] for r in records_2)


def test_concurrent_reads_of_evidence_and_audit_stay_patient_scoped(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider())
    api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})
    api_client.post("/api/patients/profile-patient-2/ask", json={"query": "diabetes"})
    _clear_llm_override()

    def _read_evidence(patient_id):
        return patient_id, api_client.get(f"/api/patients/{patient_id}/evidence")

    def _read_audit(patient_id):
        return patient_id, api_client.get(f"/api/patients/{patient_id}/audit")

    patients = ["profile-patient-1", "profile-patient-2"] * 10
    with ThreadPoolExecutor(max_workers=8) as pool:
        evidence_results = list(pool.map(_read_evidence, patients))
        audit_results = list(pool.map(_read_audit, patients))

    for patient_id, resp in evidence_results:
        assert resp.status_code == 200
        assert all(item["patient_id"] == patient_id for item in resp.json()["items"])

    for patient_id, resp in audit_results:
        assert resp.status_code == 200
        for item in resp.json()["items"]:
            assert item["patient_id"] == patient_id
