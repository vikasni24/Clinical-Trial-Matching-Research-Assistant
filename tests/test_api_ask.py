"""Phase 6G: POST /api/patients/{patient_id}/ask — the grounded
question-answering pipeline exposed through one endpoint. Every test here
overrides the LLM provider dependency with a deterministic, offline fake;
a real LLM is never called from automated tests."""

import shutil

from app.api.routes.patients import get_llm_provider
from app.main import app
from app.services.anthropic_llm_provider import LLMProviderRequestError
from app.services.fhir_ingestion import FHIRIngestionService
from app.services.patient_normalization import PatientNormalizationService


def _setup(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()
    PatientNormalizationService(mongo_db).normalize_all()


class _FakeLLMProvider:
    """A deterministic, offline stand-in for a real LLMProvider. Records
    every prompt it's asked to generate from, so tests can assert whether
    (and with what content) it was ever actually invoked."""

    def __init__(self, response: str = "The patient appears generally healthy."):
        self.calls: list[str] = []
        self._response = response

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


class _RaisingLLMProvider:
    def __init__(self, exc: Exception):
        self.calls: list[str] = []
        self._exc = exc

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        raise self._exc


def _override_llm(fake):
    app.dependency_overrides[get_llm_provider] = lambda: fake


def _clear_llm_override():
    app.dependency_overrides.pop(get_llm_provider, None)


# --- grounded answer (200, status "answered") ---------------------------------------


def test_ask_returns_grounded_answer_for_evidence_backed_query(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    fake = _FakeLLMProvider("The patient has a recorded diagnosis of hypertension [Condition/cond-1].")
    _override_llm(fake)

    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})

    _clear_llm_override()
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "answered"
    assert body["patient_id"] == "profile-patient-1"
    assert body["query"] == "hypertension"
    assert body["evidence"][0]["resource_id"] == "cond-1"
    assert len(fake.calls) == 1


def test_ask_prompt_never_contains_raw_fhir(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    fake = _FakeLLMProvider("The patient has hypertension [Condition/cond-1].")
    _override_llm(fake)

    api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})

    _clear_llm_override()
    assert len(fake.calls) == 1
    prompt_text = fake.calls[0]
    for marker in ("resourceType", "clinicalStatus", "valueQuantity", '"_id"', "fhir_resources"):
        assert marker not in prompt_text


# --- unsupported query (never invented as a negative fact, no LLM call) -------------


def test_ask_unsupported_query_returns_unsupported_without_calling_llm(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    fake = _FakeLLMProvider()
    _override_llm(fake)

    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "???"})

    _clear_llm_override()
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "unsupported"
    assert body["answer_text"] is None
    assert body["evidence"] == []
    assert fake.calls == []


# --- insufficient evidence: missing evidence never becomes a negative fact ----------


def test_ask_no_evidence_returns_insufficient_evidence_without_calling_llm(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    fake = _FakeLLMProvider()
    _override_llm(fake)

    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hemoglobin a1c"})

    _clear_llm_override()
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "insufficient_evidence"
    assert body["answer_text"] is None
    assert body["evidence"] == []
    assert fake.calls == []
    # Missing evidence is never phrased as a negative clinical fact.
    assert "does not have" not in (body["message"] or "").lower()
    assert "negative" not in (body["message"] or "").lower()


def test_ask_evidence_found_but_uncited_answer_is_insufficient_evidence(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    fake = _FakeLLMProvider("The patient appears generally healthy.")
    _override_llm(fake)

    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})

    _clear_llm_override()
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "insufficient_evidence"
    assert body["answer_text"] is None
    assert len(fake.calls) == 1  # evidence existed, so generation WAS attempted


def test_ask_never_invents_a_fabricated_evidence_reference(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    fake = _FakeLLMProvider("See [Condition/does-not-exist] for details.")
    _override_llm(fake)

    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})

    _clear_llm_override()
    body = resp.json()
    assert body["status"] == "insufficient_evidence"
    assert body["evidence"] == []


# --- unknown patient (validated before retrieval) -----------------------------------


def test_ask_unknown_patient_returns_404(api_client):
    resp = api_client.post("/api/patients/does-not-exist/ask", json={"query": "hypertension"})

    assert resp.status_code == 404


def test_ask_unknown_patient_never_calls_llm(api_client):
    fake = _FakeLLMProvider()
    _override_llm(fake)

    api_client.post("/api/patients/does-not-exist/ask", json={"query": "hypertension"})

    _clear_llm_override()
    assert fake.calls == []


# --- empty query -----------------------------------------------------------------------


def test_ask_empty_query_is_rejected(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": ""})

    assert resp.status_code in (400, 422)


def test_ask_whitespace_only_query_is_rejected(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "   "})

    assert resp.status_code == 400


# --- provider failure ------------------------------------------------------------------


def test_ask_llm_provider_failure_returns_502(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_RaisingLLMProvider(LLMProviderRequestError("simulated provider outage")))

    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})

    _clear_llm_override()
    assert resp.status_code == 502


# --- patient isolation -------------------------------------------------------------------


def test_ask_never_retrieves_another_patients_evidence(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    fake = _FakeLLMProvider("The patient has diabetes [Condition/cond-2].")
    _override_llm(fake)

    resp = api_client.post("/api/patients/profile-patient-2/ask", json={"query": "diabetes"})

    _clear_llm_override()
    body = resp.json()
    assert body["patient_id"] == "profile-patient-2"
    assert all(item["patient_id"] == "profile-patient-2" for item in body["evidence"])
    assert "cond-1" not in [item["resource_id"] for item in body["evidence"]]


# --- determinism -------------------------------------------------------------------------


def test_ask_repeated_identical_requests_are_deterministic(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider("The patient has hypertension [Condition/cond-1]."))

    first = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"}).json()
    second = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"}).json()

    _clear_llm_override()
    assert first == second


# --- no raw FHIR / no MongoDB _id in the response -----------------------------------------


def test_ask_response_never_contains_raw_fhir_or_mongo_id(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider("The patient has hypertension [Condition/cond-1]."))

    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})

    _clear_llm_override()
    body = resp.json()
    assert "data" not in body
    for item in body["evidence"]:
        assert "data" not in item
        assert "resourceType" not in item
        assert "_id" not in item
