"""Phase 5D: deterministic, dependency-free semantic-ish retrieval. Uses
mongomock for patient-scoped tests — no network, no external model, no
credentials required anywhere in this module or the code under test."""

import inspect

import pytest

from app.models.evidence import Evidence
from app.models.retrieval import RetrievalRequest, RetrievalResult
from app.services.semantic_retriever import DEFAULT_LIMIT, MIN_RELEVANCE_THRESHOLD, SemanticEvidenceRetriever


def _insert_observation(mongo_db, patient_id, resource_id, code, display, value, unit, effective_date="2023-01-10"):
    mongo_db["fhir_resources"].insert_one(
        {
            "patient_id": patient_id,
            "resource_type": "Observation",
            "resource_id": resource_id,
            "data": {
                "resourceType": "Observation",
                "id": resource_id,
                "status": "final",
                "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}], "text": display},
                "effectiveDateTime": effective_date,
                "valueQuantity": {"value": value, "unit": unit},
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


# --- 1: implements EvidenceRetriever --------------------------------------------


def test_implements_evidence_retriever_protocol(mongo_db):
    retriever = SemanticEvidenceRetriever(mongo_db)
    assert hasattr(retriever, "retrieve")

    # Called exactly as the Phase 5A protocol specifies — no extra args.
    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="blood pressure"))
    assert isinstance(result, RetrievalResult)


# --- 2-3: relevant evidence retrieved and ranked above unrelated ---------------


def test_natural_language_query_retrieves_relevant_evidence(mongo_db):
    _insert_observation(mongo_db, "p1", "obs-bp", "8480-6", "Systolic Blood Pressure", 125, "mm[Hg]")
    retriever = SemanticEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="What is the patient's blood pressure?"))

    assert result.status == "evidence_found"
    assert result.evidence[0].resource_id == "obs-bp"


def test_relevant_evidence_ranks_above_unrelated_evidence(mongo_db):
    _insert_observation(mongo_db, "p1", "obs-bp", "8480-6", "Systolic Blood Pressure", 125, "mm[Hg]")
    _insert_condition(mongo_db, "p1", "cond-htn", "59621000", "Hypertension")
    retriever = SemanticEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="blood pressure reading"))

    assert result.status == "evidence_found"
    result_ids = [e.resource_id for e in result.evidence]
    assert "obs-bp" in result_ids
    # Hypertension shares no vocabulary with "blood pressure reading" — it
    # must not outrank (or, given the threshold, even appear alongside) the
    # genuinely relevant Observation.
    assert result_ids[0] == "obs-bp"


# --- 4-5: top-K -------------------------------------------------------------------


def test_top_k_is_enforced(mongo_db):
    for i in range(8):
        _insert_observation(mongo_db, "p1", f"obs-{i}", "8480-6", "Systolic Blood Pressure", 120 + i, "mm[Hg]")
    retriever = SemanticEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="blood pressure"), limit=3)

    assert len(result.evidence) == 3


def test_default_limit_is_five(mongo_db):
    for i in range(8):
        _insert_observation(mongo_db, "p1", f"obs-{i}", "8480-6", "Systolic Blood Pressure", 120 + i, "mm[Hg]")
    retriever = SemanticEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="blood pressure"))

    assert DEFAULT_LIMIT == 5
    assert len(result.evidence) == 5


def test_invalid_limit_is_rejected(mongo_db):
    retriever = SemanticEvidenceRetriever(mongo_db)

    with pytest.raises(ValueError):
        retriever.retrieve(RetrievalRequest(patient_id="p1", query="blood pressure"), limit=0)

    with pytest.raises(ValueError):
        retriever.retrieve(RetrievalRequest(patient_id="p1", query="blood pressure"), limit=-1)


# --- 6-7: no-evidence / unsupported safety --------------------------------------


def test_no_evidence_returns_no_evidence_found(mongo_db):
    _insert_observation(mongo_db, "p1", "obs-glucose", "2339-0", "Glucose", 90, "mg/dL")
    retriever = SemanticEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="hemoglobin a1c level"))

    assert result.status == "no_evidence_found"
    assert result.evidence == []


def test_meaningless_query_is_unsupported_not_guessed(mongo_db):
    _insert_observation(mongo_db, "p1", "obs-bp", "8480-6", "Systolic Blood Pressure", 125, "mm[Hg]")
    retriever = SemanticEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="???"))

    assert result.status == "unsupported"
    assert result.evidence == []
    assert result.message is not None


# --- 8-9, 20: patient isolation / unknown patient / no cross-patient leakage ----


def test_patient_isolation(mongo_db):
    _insert_observation(mongo_db, "patient-a", "obs-a", "8480-6", "Systolic Blood Pressure", 125, "mm[Hg]")
    _insert_observation(mongo_db, "patient-b", "obs-b", "8480-6", "Systolic Blood Pressure", 140, "mm[Hg]")
    retriever = SemanticEvidenceRetriever(mongo_db)

    result_a = retriever.retrieve(RetrievalRequest(patient_id="patient-a", query="blood pressure"))
    result_b = retriever.retrieve(RetrievalRequest(patient_id="patient-b", query="blood pressure"))

    assert [e.resource_id for e in result_a.evidence] == ["obs-a"]
    assert [e.resource_id for e in result_b.evidence] == ["obs-b"]
    assert all(e.patient_id == "patient-a" for e in result_a.evidence)
    assert all(e.patient_id == "patient-b" for e in result_b.evidence)


def test_unknown_patient_returns_no_evidence_found_not_an_error(mongo_db):
    retriever = SemanticEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="does-not-exist", query="blood pressure"))

    assert result.status == "no_evidence_found"
    assert result.evidence == []


def test_no_cross_patient_semantic_index_leakage(mongo_db):
    # Patient B has abundant, highly relevant evidence; patient A has none.
    # A must never receive B's evidence merely because B's is a strong match.
    for i in range(5):
        _insert_observation(mongo_db, "patient-b", f"obs-b-{i}", "8480-6", "Systolic Blood Pressure", 120 + i, "mm[Hg]")
    retriever = SemanticEvidenceRetriever(mongo_db)

    result_a = retriever.retrieve(RetrievalRequest(patient_id="patient-a", query="blood pressure"))

    assert result_a.status == "no_evidence_found"
    assert result_a.evidence == []


# --- 10-11: determinism / tie-breaking ------------------------------------------


def test_deterministic_repeated_retrieval(mongo_db):
    _insert_observation(mongo_db, "p1", "obs-bp", "8480-6", "Systolic Blood Pressure", 125, "mm[Hg]")
    retriever = SemanticEvidenceRetriever(mongo_db)
    request = RetrievalRequest(patient_id="p1", query="blood pressure")

    first = retriever.retrieve(request)
    second = retriever.retrieve(request)

    assert first.model_dump() == second.model_dump()


def test_tie_breaking_is_deterministic():
    # Two evidence items with identical indexed text (and therefore
    # identical similarity to any query) must still order consistently.
    from app.services.semantic_retriever import _cosine_similarity, _evidence_text, _tokenize

    e1 = Evidence(patient_id="p1", resource_type="Observation", resource_id="obs-z", code="1", display="Glucose", value=90)
    e2 = Evidence(patient_id="p1", resource_type="Observation", resource_id="obs-a", code="1", display="Glucose", value=90)

    query_vector = _tokenize("glucose")
    sim1 = _cosine_similarity(query_vector, _tokenize(_evidence_text(e1)))
    sim2 = _cosine_similarity(query_vector, _tokenize(_evidence_text(e2)))
    assert sim1 == sim2 > 0  # confirm this is genuinely a tie before checking order


def test_tie_breaking_orders_identical_similarity_items_stably(mongo_db):
    # Two Observations with identical clinically-descriptive text (so
    # identical similarity to the query) but different resource_ids.
    _insert_observation(mongo_db, "p1", "obs-z", "2339-0", "Glucose", 90, "mg/dL", effective_date="2023-01-01")
    _insert_observation(mongo_db, "p1", "obs-a", "2339-0", "Glucose", 90, "mg/dL", effective_date="2023-01-01")
    retriever = SemanticEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="glucose"))

    assert [e.resource_id for e in result.evidence] == ["obs-a", "obs-z"]


# --- 12-14: no raw FHIR / no _id / plain Evidence objects -----------------------


def test_no_raw_fhir_or_mongo_id_in_result(mongo_db):
    _insert_observation(mongo_db, "p1", "obs-bp", "8480-6", "Systolic Blood Pressure", 125, "mm[Hg]")
    retriever = SemanticEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="blood pressure"))
    dumped = result.model_dump()

    assert "data" not in dumped
    for evidence_dict in dumped["evidence"]:
        assert "data" not in evidence_dict
        assert "resourceType" not in evidence_dict
        assert "_id" not in evidence_dict


def test_returned_objects_are_existing_evidence_objects(mongo_db):
    _insert_observation(mongo_db, "p1", "obs-bp", "8480-6", "Systolic Blood Pressure", 125, "mm[Hg]")
    retriever = SemanticEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="blood pressure"))

    assert all(isinstance(e, Evidence) for e in result.evidence)
    assert set(result.evidence[0].model_dump().keys()) == set(Evidence.model_fields.keys())


# --- 15: relevance threshold prevents unrelated results --------------------------


def test_relevance_threshold_excludes_unrelated_evidence(mongo_db):
    _insert_observation(mongo_db, "p1", "obs-bp", "8480-6", "Systolic Blood Pressure", 125, "mm[Hg]")
    retriever = SemanticEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="unrelated physical therapy schedule"))

    assert result.status == "no_evidence_found"
    assert result.evidence == []
    assert MIN_RELEVANCE_THRESHOLD > 0  # a real, positive cutoff exists


# --- 19: no LLM/provider API required -------------------------------------------


def test_no_llm_or_provider_api_required():
    # The module docstring legitimately names things it deliberately avoids
    # (e.g. "sentence-transformers/torch") as a negation disclaimer — so
    # this checks the actual import statements, not the docstring prose.
    import app.services.semantic_retriever as module

    imports = [
        line.strip()
        for line in inspect.getsource(module).splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    import_text = "\n".join(imports).lower()
    for forbidden in (
        "openai", "anthropic", "requests", "httpx", "urllib", "chromadb",
        "pinecone", "faiss", "torch", "sentence_transformers", "weaviate", "qdrant",
    ):
        assert forbidden not in import_text
    # Only stdlib, pymongo, and internal app modules are imported.
    for line in imports:
        assert line.startswith(("import math", "import re", "from collections", "from typing", "from __future__")) or line.startswith(
            ("from pymongo", "from app.")
        )


def test_no_confidence_or_hallucination_fields_introduced():
    import app.services.semantic_retriever as module

    source = inspect.getsource(module).lower()
    for forbidden in ("confidence_score", "clinical_probability", "hallucination_score", "ai_reasoning"):
        assert forbidden not in source
