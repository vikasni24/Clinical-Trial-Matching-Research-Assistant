"""Phase 9 Part C: patient isolation verified across the COMPLETE chain —
structured retrieval, semantic retrieval, hybrid retrieval, grounded
context, answer generation, answer validation, audit references, and the
API response — for the same two real, ingested patients in one continuous
flow. Earlier phases proved isolation at each layer independently with
hand-built objects; this file proves it holds when every layer is wired
together against real data. No real LLM is ever called."""

import shutil

from app.api.routes.patients import get_llm_provider
from app.main import app
from app.models.retrieval import RetrievalRequest
from app.repositories import audit_repository
from app.services.ask_service import AskService
from app.services.fhir_ingestion import FHIRIngestionService
from app.services.hybrid_retriever import HybridEvidenceRetriever
from app.services.patient_normalization import PatientNormalizationService
from app.services.semantic_retriever import SemanticEvidenceRetriever
from app.services.structured_retriever import StructuredEvidenceRetriever


def _setup(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()
    PatientNormalizationService(mongo_db).normalize_all()


class _FakeLLMProvider:
    def __init__(self, response: str):
        self._response = response

    def generate(self, prompt: str) -> str:
        return self._response


def _override_llm(fake):
    app.dependency_overrides[get_llm_provider] = lambda: fake


def _clear_llm_override():
    app.dependency_overrides.pop(get_llm_provider, None)


# PATIENT_1 = profile-patient-1 (Hypertension, cond-1); PATIENT_2 =
# profile-patient-2 (Diabetes, cond-2) — see tests/fixtures/profile_bundle.json.


def test_structured_retrieval_layer_isolation(mongo_db):
    # The structured retriever matches by exact registry code (see
    # structured_retriever.py's _CONCEPT_REGISTRY) — profile_bundle.json's
    # own Hypertension condition uses a different (but equally real) SNOMED
    # code, so this layer is exercised directly with registry-matching
    # synthetic data, mirroring the existing pattern in
    # tests/test_hybrid_retriever.py.
    mongo_db["fhir_resources"].insert_one(
        {
            "patient_id": "patient-a",
            "resource_type": "Condition",
            "resource_id": "cond-a",
            "data": {
                "resourceType": "Condition",
                "id": "cond-a",
                "code": {"coding": [{"system": "http://snomed.info/sct", "code": "59621000", "display": "Hypertension"}]},
            },
        }
    )
    mongo_db["fhir_resources"].insert_one(
        {
            "patient_id": "patient-b",
            "resource_type": "Condition",
            "resource_id": "cond-b",
            "data": {
                "resourceType": "Condition",
                "id": "cond-b",
                "code": {"coding": [{"system": "http://snomed.info/sct", "code": "44054006", "display": "Type 2 diabetes mellitus"}]},
            },
        }
    )
    retriever = StructuredEvidenceRetriever(mongo_db)

    result_a = retriever.retrieve(RetrievalRequest(patient_id="patient-a", query="hypertension"))
    result_b = retriever.retrieve(RetrievalRequest(patient_id="patient-b", query="hypertension"))

    assert [e.resource_id for e in result_a.evidence] == ["cond-a"]
    assert result_b.evidence == []  # patient-b has no hypertension on record
    assert all(e.patient_id == "patient-a" for e in result_a.evidence)


def test_semantic_retrieval_layer_isolation(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    retriever = SemanticEvidenceRetriever(mongo_db)

    result_1 = retriever.retrieve(RetrievalRequest(patient_id="profile-patient-1", query="blood pressure"))
    result_2 = retriever.retrieve(RetrievalRequest(patient_id="profile-patient-2", query="blood pressure"))

    assert any(e.resource_id == "obs-1" for e in result_1.evidence)
    assert all(e.patient_id == "profile-patient-1" for e in result_1.evidence)
    assert all(e.patient_id == "profile-patient-2" for e in result_2.evidence)
    assert "obs-1" not in [e.resource_id for e in result_2.evidence]


def test_hybrid_retrieval_layer_isolation(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    retriever = HybridEvidenceRetriever(mongo_db)

    result_1 = retriever.retrieve(RetrievalRequest(patient_id="profile-patient-1", query="hypertension"))
    result_2 = retriever.retrieve(RetrievalRequest(patient_id="profile-patient-2", query="diabetes"))

    assert [e.resource_id for e in result_1.evidence] == ["cond-1"]
    assert [e.resource_id for e in result_2.evidence] == ["cond-2"]
    assert all(e.patient_id == "profile-patient-1" for e in result_1.evidence)
    assert all(e.patient_id == "profile-patient-2" for e in result_2.evidence)


def test_grounded_context_layer_isolation(mongo_db, tmp_path, fixtures_dir):
    from app.services.rag_context import GroundedContextService

    _setup(mongo_db, tmp_path, fixtures_dir)
    service = GroundedContextService(mongo_db)

    context_1 = service.build_context(RetrievalRequest(patient_id="profile-patient-1", query="hypertension"))
    context_2 = service.build_context(RetrievalRequest(patient_id="profile-patient-2", query="diabetes"))

    assert context_1.patient_id == "profile-patient-1"
    assert context_2.patient_id == "profile-patient-2"
    assert all(e.patient_id == "profile-patient-1" for e in context_1.evidence)
    assert all(e.patient_id == "profile-patient-2" for e in context_2.evidence)


def test_answer_generation_and_validation_layer_isolation(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    answer_1 = AskService(mongo_db, _FakeLLMProvider("Patient has hypertension [Condition/cond-1].")).ask(
        "profile-patient-1", "hypertension"
    )
    answer_2 = AskService(mongo_db, _FakeLLMProvider("Patient has diabetes [Condition/cond-2].")).ask(
        "profile-patient-2", "diabetes"
    )

    assert answer_1.patient_id == "profile-patient-1"
    assert [e.resource_id for e in answer_1.evidence] == ["cond-1"]
    assert answer_2.patient_id == "profile-patient-2"
    assert [e.resource_id for e in answer_2.evidence] == ["cond-2"]
    # Cross-check: patient-2's evidence never appears anywhere in patient-1's answer.
    assert "cond-2" not in [e.resource_id for e in answer_1.evidence]
    assert "cond-1" not in [e.resource_id for e in answer_2.evidence]


def test_audit_reference_layer_isolation(mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    AskService(mongo_db, _FakeLLMProvider("Patient has hypertension [Condition/cond-1].")).ask(
        "profile-patient-1", "hypertension"
    )
    AskService(mongo_db, _FakeLLMProvider("Patient has diabetes [Condition/cond-2].")).ask(
        "profile-patient-2", "diabetes"
    )

    records_1, total_1 = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-1")
    records_2, total_2 = audit_repository.get_patient_audit_history(mongo_db, "profile-patient-2")

    assert total_1 == 1 and total_2 == 1
    assert records_1[0].patient_id == "profile-patient-1"
    assert [r.resource_id for r in records_1[0].evidence_references] == ["cond-1"]
    assert records_2[0].patient_id == "profile-patient-2"
    assert [r.resource_id for r in records_2[0].evidence_references] == ["cond-2"]


def test_api_layer_isolation_end_to_end(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider("Patient has hypertension [Condition/cond-1]."))
    ask_1 = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})

    _override_llm(_FakeLLMProvider("Patient has diabetes [Condition/cond-2]."))
    ask_2 = api_client.post("/api/patients/profile-patient-2/ask", json={"query": "diabetes"})

    evidence_1 = api_client.get("/api/patients/profile-patient-1/evidence")
    evidence_2 = api_client.get("/api/patients/profile-patient-2/evidence")
    audit_1 = api_client.get("/api/patients/profile-patient-1/audit")
    audit_2 = api_client.get("/api/patients/profile-patient-2/audit")
    _clear_llm_override()

    assert ask_1.json()["evidence"][0]["resource_id"] == "cond-1"
    assert ask_2.json()["evidence"][0]["resource_id"] == "cond-2"

    evidence_1_ids = {item["resource_id"] for item in evidence_1.json()["items"]}
    evidence_2_ids = {item["resource_id"] for item in evidence_2.json()["items"]}
    assert evidence_1_ids.isdisjoint(evidence_2_ids)
    assert "cond-2" not in evidence_1_ids
    assert "cond-1" not in evidence_2_ids

    assert audit_1.json()["items"][0]["patient_id"] == "profile-patient-1"
    assert audit_2.json()["items"][0]["patient_id"] == "profile-patient-2"
    assert audit_1.json()["pagination"]["total"] == 1
    assert audit_2.json()["pagination"]["total"] == 1
