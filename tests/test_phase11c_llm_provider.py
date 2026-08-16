"""Phase 11C: verifies the complete existing /ask pipeline works correctly
and safely with the existing LLMProvider abstraction — patient validation
-> HybridEvidenceRetriever -> GroundedContext -> pre-generation safety gate
-> build_grounded_prompt() -> LLMProvider.generate() -> AnswerValidator ->
GroundedAnswer -> AuditService -> API response. No new architecture, no new
retrieval strategy, no LLM/vendor SDK dependency added here — only fake
providers, exactly like every other test file in this project. Fixture
data below mirrors the exact real-data shapes confirmed against the live
MongoDB dataset in Phase 11A/11B (Acetaminophen MedicationRequest with
status="stopped"; a Blood Pressure panel Observation with component[]
systolic/diastolic)."""

import shutil

import pytest

from app.api.routes.patients import get_llm_provider
from app.main import app
from app.models.retrieval import RetrievalRequest
from app.repositories import audit_repository
from app.services.anthropic_llm_provider import LLMProviderConfigurationError, LLMProviderRequestError
from app.services.ask_service import AskService
from app.services.fhir_ingestion import FHIRIngestionService
from app.services.grounded_prompt import build_grounded_prompt
from app.services.patient_normalization import PatientNormalizationService
from app.services.rag_context import GroundedContextService


# --- fixtures: realistic, mirroring the actual real-dataset shapes found in Phase 11A/11B ------


def _insert_medication(mongo_db, patient_id, resource_id, code, display, status="stopped", authored_on="2012-11-30"):
    mongo_db["fhir_resources"].insert_one(
        {
            "patient_id": patient_id,
            "resource_type": "MedicationRequest",
            "resource_id": resource_id,
            "data": {
                "resourceType": "MedicationRequest",
                "id": resource_id,
                "status": status,
                "intent": "order",
                "medicationCodeableConcept": {
                    "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": code, "display": display}],
                    "text": display,
                },
                "subject": {"reference": f"Patient/{patient_id}"},
                "authoredOn": authored_on,
            },
        }
    )


def _insert_bp_observation(mongo_db, patient_id, resource_id, systolic, diastolic, effective_date="2023-01-10"):
    mongo_db["fhir_resources"].insert_one(
        {
            "patient_id": patient_id,
            "resource_type": "Observation",
            "resource_id": resource_id,
            "data": {
                "resourceType": "Observation",
                "id": resource_id,
                "status": "final",
                "code": {"coding": [{"system": "http://loinc.org", "code": "85354-9", "display": "Blood Pressure"}], "text": "Blood Pressure"},
                "effectiveDateTime": effective_date,
                "component": [
                    {
                        "code": {"coding": [{"system": "http://loinc.org", "code": "8462-4", "display": "Diastolic Blood Pressure"}]},
                        "valueQuantity": {"value": diastolic, "unit": "mm[Hg]"},
                    },
                    {
                        "code": {"coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic Blood Pressure"}]},
                        "valueQuantity": {"value": systolic, "unit": "mm[Hg]"},
                    },
                ],
            },
        }
    )


def _insert_condition(mongo_db, patient_id, resource_id, code, display):
    mongo_db["fhir_resources"].insert_one(
        {
            "patient_id": patient_id,
            "resource_type": "Condition",
            "resource_id": resource_id,
            "data": {
                "resourceType": "Condition",
                "id": resource_id,
                "code": {"coding": [{"system": "http://snomed.info/sct", "code": code, "display": display}], "text": display},
                "clinicalStatus": {"coding": [{"code": "active"}]},
            },
        }
    )


def _setup_realistic_patient(mongo_db, patient_id):
    _insert_medication(mongo_db, patient_id, "med-1", "313782", "Acetaminophen 325 MG Oral Tablet", status="stopped")
    _insert_bp_observation(mongo_db, patient_id, "bp-1", systolic=125, diastolic=85)


def _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir):
    """The shared fixture bundle used across the suite — real patients
    profile-patient-1/profile-patient-2 with real, ingested FHIR data."""
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()
    PatientNormalizationService(mongo_db).normalize_all()


class _FakeLLMProvider:
    def __init__(self, response: str = "unused"):
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


# =====================================================================
# Item 1-2 — missing API key / provider configuration error
# =====================================================================


def test_missing_api_key_raises_configuration_error_not_request_error(mongo_db):
    from app.config import Settings
    from app.services.anthropic_llm_provider import AnthropicLLMProvider

    with pytest.raises(LLMProviderConfigurationError) as exc_info:
        AnthropicLLMProvider(settings=Settings(llm_api_key=None))

    # Static, secret-free message — never embeds a (nonexistent) key value.
    assert "LLM_API_KEY" in str(exc_info.value)


def test_ask_service_propagates_configuration_error_unmodified(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    class _MisconfiguredProvider:
        def generate(self, prompt: str) -> str:
            raise LLMProviderConfigurationError("LLM_API_KEY is not configured")

    with pytest.raises(LLMProviderConfigurationError):
        AskService(mongo_db, _MisconfiguredProvider()).ask("profile-patient-1", "hypertension")


# =====================================================================
# Item 3-4 — provider request error / safe error returned by the API
# =====================================================================


def test_provider_request_error_is_distinct_from_configuration_error(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    class _FailingProvider:
        def generate(self, prompt: str) -> str:
            raise LLMProviderRequestError("simulated network failure")

    with pytest.raises(LLMProviderRequestError):
        AskService(mongo_db, _FailingProvider()).ask("profile-patient-1", "hypertension")


def test_api_returns_generic_502_never_the_raw_upstream_body(api_client, mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)
    sensitive_upstream_body = '{"error": "invalid_request", "trace_id": "SENSITIVE-UPSTREAM-abc123"}'

    class _FailingProvider:
        def generate(self, prompt: str) -> str:
            raise LLMProviderRequestError(f"LLM provider returned an error response: 400 {sensitive_upstream_body}")

    _override_llm(_FailingProvider())
    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})
    _clear_llm_override()

    assert resp.status_code == 502
    assert resp.json() == {"detail": "LLM provider request failed"}
    assert "SENSITIVE-UPSTREAM-abc123" not in resp.text


def test_api_returns_500_for_configuration_error_with_safe_message(api_client, mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    class _MisconfiguredProvider:
        def generate(self, prompt: str) -> str:
            raise LLMProviderConfigurationError("LLM_API_KEY is not configured")

    _override_llm(_MisconfiguredProvider())
    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})
    _clear_llm_override()

    assert resp.status_code == 500
    assert "LLM_API_KEY" in resp.json()["detail"]
    # No secret value, header, or stack trace of any kind in the body.
    for forbidden in ("sk-", "sk-ant-", "Authorization", "Bearer", "Traceback"):
        assert forbidden not in resp.text


# =====================================================================
# TEST A / items 5-6 — successful generation + valid citation accepted
# =====================================================================


def test_a_medication_question_returns_answered_with_only_grounded_context_evidence(mongo_db):
    _setup_realistic_patient(mongo_db, "p1")
    provider = _FakeLLMProvider("A recorded MedicationRequest is Acetaminophen 325 MG Oral Tablet [MedicationRequest/med-1].")

    answer = AskService(mongo_db, provider).ask("p1", "What medications is the patient taking?")

    assert answer.status == "answered"
    assert len(provider.calls) == 1
    assert [e.resource_id for e in answer.evidence] == ["med-1"]
    # Every returned Evidence object must have originated from GroundedContext,
    # never invented — cross-check against retrieval performed independently.
    context = GroundedContextService(mongo_db).build_context(
        RetrievalRequest(patient_id="p1", query="What medications is the patient taking?")
    )
    context_ids = {e.resource_id for e in context.evidence}
    assert {e.resource_id for e in answer.evidence}.issubset(context_ids)


# =====================================================================
# TEST B — blood pressure, component-derived value
# =====================================================================


def test_b_blood_pressure_question_returns_answered_with_component_derived_value(mongo_db):
    _setup_realistic_patient(mongo_db, "p1")
    provider = _FakeLLMProvider("Blood pressure was 125/85 [Observation/bp-1].")

    answer = AskService(mongo_db, provider).ask("p1", "What is the patient's blood pressure?")

    assert answer.status == "answered"
    assert answer.evidence[0].resource_id == "bp-1"
    value_text = answer.evidence[0].value
    assert "Systolic Blood Pressure" in value_text
    assert "Diastolic Blood Pressure" in value_text
    assert "125" in value_text
    assert "85" in value_text


# =====================================================================
# TEST C / item 10 — no evidence, LLM never called
# =====================================================================


def test_c_no_hba1c_evidence_is_insufficient_evidence_without_calling_llm(mongo_db):
    _setup_realistic_patient(mongo_db, "p1")  # real evidence exists, just not HbA1c
    provider = _FakeLLMProvider()

    answer = AskService(mongo_db, provider).ask("p1", "What is the patient's HbA1c?")

    assert answer.status == "insufficient_evidence"
    assert provider.calls == []
    assert answer.answer_text is None
    assert answer.evidence == []


# =====================================================================
# TEST D / item 9 — unsupported query, LLM never called
# =====================================================================


def test_d_unsupported_query_returns_unsupported_without_calling_llm(mongo_db):
    _setup_realistic_patient(mongo_db, "p1")
    provider = _FakeLLMProvider()

    answer = AskService(mongo_db, provider).ask("p1", "???")

    assert answer.status == "unsupported"
    assert provider.calls == []


# =====================================================================
# TEST E / item 7 — fabricated citation rejected
# =====================================================================


def test_e_fabricated_citation_never_reaches_the_final_response(mongo_db):
    _setup_realistic_patient(mongo_db, "p1")
    provider = _FakeLLMProvider("The patient also takes [MedicationRequest/does-not-exist].")

    answer = AskService(mongo_db, provider).ask("p1", "What medications is the patient taking?")

    assert answer.status == "insufficient_evidence"
    assert answer.evidence == []
    assert answer.answer_text is None
    assert "does-not-exist" not in [e.resource_id for e in answer.evidence]


# =====================================================================
# TEST F / item 8 — uncited confident answer rejected
# =====================================================================


def test_f_uncited_confident_answer_is_rejected(mongo_db):
    _setup_realistic_patient(mongo_db, "p1")
    provider = _FakeLLMProvider("The patient is taking Acetaminophen.")  # no citation at all

    answer = AskService(mongo_db, provider).ask("p1", "What medications is the patient taking?")

    assert answer.status == "insufficient_evidence"
    assert answer.evidence == []
    assert answer.answer_text is None


# =====================================================================
# Grounded prompt inspection (Step 6)
# =====================================================================


def test_grounded_prompt_contains_no_raw_fhir_or_mongo_id(mongo_db):
    _setup_realistic_patient(mongo_db, "p1")
    context = GroundedContextService(mongo_db).build_context(
        RetrievalRequest(patient_id="p1", query="What medications is the patient taking?")
    )

    prompt = build_grounded_prompt(context)
    full_text = "\n".join([prompt.instructions, prompt.status_note, prompt.evidence_text])

    for forbidden in ("resourceType", "medicationCodeableConcept", '"_id"', "fhir_resources", "subject"):
        assert forbidden not in full_text


def test_grounded_prompt_contains_no_secrets(mongo_db):
    import httpx

    from app.config import Settings
    from app.services.anthropic_llm_provider import AnthropicLLMProvider

    fake_key = "sk-ant-super-secret-test-key-should-never-leak"
    settings = Settings(llm_api_key=fake_key)
    provider = AnthropicLLMProvider(settings=settings, client=httpx.Client())
    assert provider._api_key == fake_key  # the provider does hold it, as expected

    # build_grounded_prompt has no access to Settings/credentials at all —
    # confirm the actual rendered prompt text for a real context never
    # contains the key, regardless of what provider/settings exist
    # elsewhere in the process.
    _setup_realistic_patient(mongo_db, "p1")
    context = GroundedContextService(mongo_db).build_context(
        RetrievalRequest(patient_id="p1", query="What medications is the patient taking?")
    )
    prompt = build_grounded_prompt(context)
    full_text = "\n".join([prompt.instructions, prompt.status_note, prompt.evidence_text])
    assert fake_key not in full_text


def test_grounded_prompt_includes_enough_evidence_to_answer_and_cite():
    from app.models.evidence import Evidence
    from app.models.rag_context import GroundedContext

    context = GroundedContext(
        patient_id="p1",
        query="What medications is the patient taking?",
        status="evidence_found",
        evidence=[
            Evidence(
                patient_id="p1", resource_type="MedicationRequest", resource_id="med-1",
                display="Acetaminophen 325 MG Oral Tablet", status="stopped",
            )
        ],
    )

    prompt = build_grounded_prompt(context)

    assert "[MedicationRequest/med-1]" in prompt.evidence_text
    assert "Acetaminophen 325 MG Oral Tablet" in prompt.evidence_text
    assert "stopped" in prompt.evidence_text
    # Grounding instructions remain intact and unweakened.
    assert "traceable to a specific evidence item" in prompt.instructions
    assert "Do not fabricate citations" in prompt.instructions


# =====================================================================
# Patient isolation through the full pipeline (Step 7 / item 11)
# =====================================================================


def test_patient_isolation_through_retrieval_context_prompt_and_answer(mongo_db):
    _insert_medication(mongo_db, "patient-a", "med-a", "313782", "Acetaminophen 325 MG Oral Tablet")
    _insert_medication(mongo_db, "patient-b", "med-b", "860975", "Metformin 500 MG Oral Tablet")

    # 1. Retrieval
    context_a = GroundedContextService(mongo_db).build_context(
        RetrievalRequest(patient_id="patient-a", query="What medications is the patient taking?")
    )
    context_b = GroundedContextService(mongo_db).build_context(
        RetrievalRequest(patient_id="patient-b", query="What medications is the patient taking?")
    )
    assert [e.resource_id for e in context_a.evidence] == ["med-a"]
    assert [e.resource_id for e in context_b.evidence] == ["med-b"]

    # 2. Rendered prompt text
    prompt_a = build_grounded_prompt(context_a)
    prompt_b = build_grounded_prompt(context_b)
    assert "med-b" not in prompt_a.evidence_text and "Metformin" not in prompt_a.evidence_text
    assert "med-a" not in prompt_b.evidence_text and "Acetaminophen" not in prompt_b.evidence_text

    # 3. What the fake provider actually receives ("fake provider input")
    provider = _FakeLLMProvider()
    provider.generate("\n\n".join([prompt_a.instructions, prompt_a.status_note, prompt_a.evidence_text]))
    provider.generate("\n\n".join([prompt_b.instructions, prompt_b.status_note, prompt_b.evidence_text]))
    assert "med-b" not in provider.calls[0]
    assert "med-a" not in provider.calls[1]

    # 4. Final GroundedAnswer
    answer_a = AskService(mongo_db, _FakeLLMProvider("Noted [MedicationRequest/med-a].")).ask(
        "patient-a", "What medications is the patient taking?"
    )
    answer_b = AskService(mongo_db, _FakeLLMProvider("Noted [MedicationRequest/med-b].")).ask(
        "patient-b", "What medications is the patient taking?"
    )
    assert [e.resource_id for e in answer_a.evidence] == ["med-a"]
    assert [e.resource_id for e in answer_b.evidence] == ["med-b"]

    # 5. Audit record
    records_a, _ = audit_repository.get_patient_audit_history(mongo_db, "patient-a")
    records_b, _ = audit_repository.get_patient_audit_history(mongo_db, "patient-b")
    assert [r.resource_id for r in records_a[0].evidence_references] == ["med-a"]
    assert [r.resource_id for r in records_b[0].evidence_references] == ["med-b"]
    assert records_a[0].patient_id == "patient-a"
    assert records_b[0].patient_id == "patient-b"


def test_patient_isolation_holds_in_the_reverse_order(mongo_db):
    _insert_medication(mongo_db, "patient-a", "med-a", "313782", "Acetaminophen 325 MG Oral Tablet")
    _insert_medication(mongo_db, "patient-b", "med-b", "860975", "Metformin 500 MG Oral Tablet")

    answer_b_first = AskService(mongo_db, _FakeLLMProvider("Noted [MedicationRequest/med-b].")).ask(
        "patient-b", "What medications is the patient taking?"
    )
    answer_a_second = AskService(mongo_db, _FakeLLMProvider("Noted [MedicationRequest/med-a].")).ask(
        "patient-a", "What medications is the patient taking?"
    )

    assert [e.resource_id for e in answer_b_first.evidence] == ["med-b"]
    assert [e.resource_id for e in answer_a_second.evidence] == ["med-a"]


# =====================================================================
# Audit integration (Step 8 / items 12-15)
# =====================================================================


def test_successful_answer_creates_appropriate_audit_record(mongo_db):
    _setup_realistic_patient(mongo_db, "p1")
    AskService(mongo_db, _FakeLLMProvider("Noted [MedicationRequest/med-1].")).ask(
        "p1", "What medications is the patient taking?"
    )

    records, total = audit_repository.get_patient_audit_history(mongo_db, "p1")
    assert total == 1
    assert records[0].answer_status == "answered"
    assert records[0].retrieval_status == "evidence_found"
    assert [r.resource_id for r in records[0].evidence_references] == ["med-1"]


def test_insufficient_evidence_response_is_correctly_auditable(mongo_db):
    _setup_realistic_patient(mongo_db, "p1")
    AskService(mongo_db, _FakeLLMProvider()).ask("p1", "What is the patient's HbA1c?")

    records, total = audit_repository.get_patient_audit_history(mongo_db, "p1")
    assert total == 1
    assert records[0].answer_status == "insufficient_evidence"
    assert records[0].retrieval_status == "no_evidence_found"
    assert records[0].evidence_references == []


def test_unsupported_response_is_correctly_auditable(mongo_db):
    _setup_realistic_patient(mongo_db, "p1")
    AskService(mongo_db, _FakeLLMProvider()).ask("p1", "???")

    records, total = audit_repository.get_patient_audit_history(mongo_db, "p1")
    assert total == 1
    assert records[0].answer_status == "unsupported"
    assert records[0].retrieval_status == "unsupported"


def test_llm_failure_creates_no_audit_record(mongo_db):
    _setup_realistic_patient(mongo_db, "p1")
    provider = _RaisingLLMProvider(LLMProviderRequestError("simulated failure"))

    with pytest.raises(LLMProviderRequestError):
        AskService(mongo_db, provider).ask("p1", "What medications is the patient taking?")

    _, total = audit_repository.get_patient_audit_history(mongo_db, "p1")
    assert total == 0


def test_audit_persistence_failure_does_not_change_an_already_validated_answer(mongo_db, monkeypatch):
    _setup_realistic_patient(mongo_db, "p1")
    provider = _FakeLLMProvider("Noted [MedicationRequest/med-1].")

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated audit write failure")

    monkeypatch.setattr(audit_repository, "insert_audit_record", _raise)

    answer = AskService(mongo_db, provider).ask("p1", "What medications is the patient taking?")

    assert answer.status == "answered"
    assert answer.evidence[0].resource_id == "med-1"


def test_audit_record_never_contains_raw_llm_output_or_prompt(mongo_db):
    raw_output = "Noted [MedicationRequest/med-1]. INTERNAL-CHAIN-OF-THOUGHT-MARKER step by step..."
    _setup_realistic_patient(mongo_db, "p1")
    AskService(mongo_db, _FakeLLMProvider(raw_output)).ask("p1", "What medications is the patient taking?")

    stored = mongo_db["audit_records"].find_one({"patient_id": "p1"})
    assert stored is not None
    for forbidden_key in ("answer_text", "prompt", "raw_llm_output", "data", "resourceType"):
        assert forbidden_key not in stored
    assert "INTERNAL-CHAIN-OF-THOUGHT-MARKER" not in str(stored)


def test_audit_record_never_contains_an_api_key(mongo_db):
    _setup_realistic_patient(mongo_db, "p1")
    AskService(mongo_db, _FakeLLMProvider("Noted [MedicationRequest/med-1].")).ask(
        "p1", "What medications is the patient taking?"
    )

    stored = mongo_db["audit_records"].find_one({"patient_id": "p1"})
    for forbidden in ("sk-", "sk-ant-", "api_key", "Authorization", "Bearer"):
        assert forbidden not in str(stored)


# =====================================================================
# API verification (Step 10)
# =====================================================================


def test_api_valid_evidence_backed_question(api_client, mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider("The patient has hypertension [Condition/cond-1]."))

    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})

    _clear_llm_override()
    assert resp.status_code == 200
    assert resp.json()["status"] == "answered"


def test_api_no_evidence_question(api_client, mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider())

    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hemoglobin a1c"})

    _clear_llm_override()
    assert resp.status_code == 200
    assert resp.json()["status"] == "insufficient_evidence"


def test_api_unsupported_question(api_client, mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider())

    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "???"})

    _clear_llm_override()
    assert resp.status_code == 200
    assert resp.json()["status"] == "unsupported"


def test_api_invalid_patient(api_client):
    resp = api_client.post("/api/patients/does-not-exist/ask", json={"query": "hypertension"})

    assert resp.status_code == 404


def test_api_empty_query(api_client, mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "   "})

    assert resp.status_code == 400


def test_api_provider_configuration_failure(api_client, mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    class _MisconfiguredProvider:
        def generate(self, prompt: str) -> str:
            raise LLMProviderConfigurationError("LLM_API_KEY is not configured")

    _override_llm(_MisconfiguredProvider())
    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})
    _clear_llm_override()

    assert resp.status_code == 500


def test_api_provider_request_failure(api_client, mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    class _FailingProvider:
        def generate(self, prompt: str) -> str:
            raise LLMProviderRequestError("simulated network failure")

    _override_llm(_FailingProvider())
    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})
    _clear_llm_override()

    assert resp.status_code == 502


# =====================================================================
# Provider-agnostic boundary (Step 2)
# =====================================================================


def test_ask_service_module_imports_no_vendor_sdk():
    import inspect
    import app.services.ask_service as module
    import app.services.grounded_prompt as prompt_module
    import app.services.safety_rules as safety_module

    for mod in (module, prompt_module, safety_module):
        source = inspect.getsource(mod)
        for forbidden in ("import httpx", "import anthropic", "from anthropic", "anthropic_llm_provider"):
            assert forbidden not in source


def test_ask_service_accepts_any_duck_typed_provider(mongo_db):
    _setup_realistic_patient(mongo_db, "p1")

    class _ArbitraryProvider:
        def generate(self, prompt: str) -> str:
            return "Noted [MedicationRequest/med-1]."

    answer = AskService(mongo_db, _ArbitraryProvider()).ask("p1", "What medications is the patient taking?")

    assert answer.status == "answered"
