"""Phase 5A: the retrieval CONTRACT only — no retrieval strategy exists
yet. These tests exercise the Pydantic models directly (no MongoDB, no
fixtures) since this phase is pure interface/data-shape design."""

import pytest
from pydantic import ValidationError

from app.models.evidence import Evidence
from app.models.retrieval import RetrievalRequest, RetrievalResult
from app.services.retrieval import EvidenceRetriever


def _evidence(patient_id: str = "p1", resource_id: str = "obs-1") -> Evidence:
    return Evidence(
        patient_id=patient_id,
        resource_type="Observation",
        resource_id=resource_id,
        code="4548-4",
        coding_system="http://loinc.org",
        display="Hemoglobin A1c",
        value=5.81,
        unit="%",
        effective_date="2023-01-10",
        status="final",
    )


# --- 1-2: RetrievalRequest requires patient_id and query ---------------------


def test_retrieval_request_requires_patient_id():
    with pytest.raises(ValidationError):
        RetrievalRequest(patient_id="", query="What is the patient's HbA1c?")

    with pytest.raises(ValidationError):
        RetrievalRequest(query="What is the patient's HbA1c?")  # missing entirely


def test_retrieval_request_requires_query():
    with pytest.raises(ValidationError):
        RetrievalRequest(patient_id="p1", query="")

    with pytest.raises(ValidationError):
        RetrievalRequest(patient_id="p1")  # missing entirely


def test_valid_retrieval_request():
    request = RetrievalRequest(patient_id="p1", query="Does the patient have hypertension?")

    assert request.patient_id == "p1"
    assert request.query == "Does the patient have hypertension?"


# --- 3: result preserves the requested patient_id -----------------------------


def test_retrieval_result_preserves_patient_id():
    result = RetrievalResult.from_evidence(patient_id="p1", query="HbA1c?", evidence=[_evidence(patient_id="p1")])

    assert result.patient_id == "p1"
    assert result.query == "HbA1c?"


# --- 4 & 10: evidence references remain traceable / carried through unmodified ---


def test_evidence_references_remain_traceable():
    evidence_item = _evidence(patient_id="p1", resource_id="obs-42")
    result = RetrievalResult.from_evidence(patient_id="p1", query="HbA1c?", evidence=[evidence_item])

    carried = result.evidence[0]
    assert carried.patient_id == "p1"
    assert carried.resource_type == "Observation"
    assert carried.resource_id == "obs-42"
    assert carried.source_collection == "fhir_resources"
    assert carried.source_reference == "Observation/obs-42"


def test_existing_evidence_objects_carried_through_unmodified():
    evidence_item = _evidence()
    result = RetrievalResult.from_evidence(patient_id="p1", query="HbA1c?", evidence=[evidence_item])

    # Composition, not modification: the exact same Evidence object/values come through.
    assert result.evidence[0] == evidence_item


# --- 5: empty retrieval is explicit, never guessed -----------------------------


def test_empty_retrieval_represented_explicitly():
    result = RetrievalResult.from_evidence(patient_id="p1", query="Does the patient have a pacemaker?", evidence=[])

    assert result.status == "no_evidence_found"
    assert result.evidence == []


def test_unsupported_retrieval_is_distinct_from_empty_retrieval():
    result = RetrievalResult.unsupported(
        patient_id="p1", query="Summarize the patient's genome", message="No retrieval strategy exists for this query type yet"
    )

    assert result.status == "unsupported"
    assert result.evidence == []
    assert result.status != "no_evidence_found"  # a distinct, non-assertive state
    assert result.message == "No retrieval strategy exists for this query type yet"


def test_status_evidence_consistency_is_enforced():
    # Can't claim evidence_found with no evidence...
    with pytest.raises(ValidationError):
        RetrievalResult(patient_id="p1", query="q", status="evidence_found", evidence=[])

    # ...and can't claim no_evidence_found while carrying evidence.
    with pytest.raises(ValidationError):
        RetrievalResult(patient_id="p1", query="q", status="no_evidence_found", evidence=[_evidence(patient_id="p1")])


# --- 6-7: no AI-generated facts, no confidence/hallucination fields -----------


def test_no_llm_generated_clinical_facts():
    # RetrievalResult has no field that could hold an LLM-authored answer,
    # summary, or generated clinical text.
    prohibited = {"answer", "generated_text", "llm_output", "summary", "response_text"}
    assert prohibited.isdisjoint(RetrievalResult.model_fields.keys())


def test_no_confidence_score_or_hallucination_fields():
    prohibited = {
        "confidence_score",
        "hallucination_score",
        "ai_reasoning",
        "probability",
        "semantic_similarity",
        "relevance_score",
    }
    assert prohibited.isdisjoint(RetrievalRequest.model_fields.keys())
    assert prohibited.isdisjoint(RetrievalResult.model_fields.keys())


# --- 8: no raw FHIR document required/possible in the result ------------------


def test_no_raw_fhir_required_in_retrieval_result():
    result = RetrievalResult.from_evidence(patient_id="p1", query="HbA1c?", evidence=[_evidence()])
    dumped = result.model_dump()

    assert "data" not in dumped
    for evidence_dict in dumped["evidence"]:
        assert "data" not in evidence_dict
        assert "resourceType" not in evidence_dict
        assert "_id" not in evidence_dict


# --- 9: patient isolation is structurally enforced -----------------------------


def test_patient_isolation_is_enforced_by_the_contract():
    # A RetrievalResult literally cannot be constructed with another
    # patient's evidence attached — this is type-level, not caller discipline.
    other_patient_evidence = _evidence(patient_id="p2")

    with pytest.raises(ValidationError):
        RetrievalResult(patient_id="p1", query="q", status="evidence_found", evidence=[other_patient_evidence])

    with pytest.raises(ValidationError):
        RetrievalResult.from_evidence(patient_id="p1", query="q", evidence=[other_patient_evidence])


def test_retrieval_request_has_no_unscoped_form():
    # There is no way to express a request without a patient_id — no
    # "all patients" or optional-patient_id variant exists on the model.
    assert RetrievalRequest.model_fields["patient_id"].is_required()


# --- Protocol / interface shape -------------------------------------------------


def test_evidence_retriever_protocol_defines_retrieve_method():
    assert hasattr(EvidenceRetriever, "retrieve")


def test_evidence_retriever_protocol_has_no_concrete_implementation():
    import inspect

    import app.services.retrieval as module

    source = inspect.getsource(module)
    # No retrieval strategy — deterministic, keyword, vector, or graph — is implemented here.
    for forbidden in ("chromadb", "pinecone", "faiss", "weaviate", "qdrant", "neo4j", "openai", "anthropic", "langchain"):
        assert forbidden not in source.lower()
