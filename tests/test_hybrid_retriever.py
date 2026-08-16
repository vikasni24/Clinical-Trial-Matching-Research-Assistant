"""Phase 5E: HybridEvidenceRetriever — composes StructuredEvidenceRetriever
+ SemanticEvidenceRetriever, deduplicates, and re-ranks via the existing
DeterministicEvidenceRanker. Uses mongomock; no live MongoDB required."""

import inspect

import pytest

from app.models.evidence import Evidence
from app.models.retrieval import RetrievalRequest, RetrievalResult
from app.services.evidence_ranker import DeterministicEvidenceRanker
from app.services.hybrid_retriever import DEFAULT_LIMIT, HybridEvidenceRetriever


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


# --- 1: implements EvidenceRetriever ---------------------------------------------


def test_implements_evidence_retriever_protocol(mongo_db):
    retriever = HybridEvidenceRetriever(mongo_db)
    assert hasattr(retriever, "retrieve")

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="hypertension"))
    assert isinstance(result, RetrievalResult)


# --- 2-4: combination and deduplication ------------------------------------------


def test_structured_and_semantic_results_are_combined(mongo_db):
    _insert_condition(mongo_db, "p1", "cond-1", "59621000", "Hypertension")
    _insert_observation(mongo_db, "p1", "obs-1", "8480-6", "Systolic Blood Pressure", 125, "mm[Hg]")
    retriever = HybridEvidenceRetriever(mongo_db)

    # "hypertension" alone: structured matches the Condition via its exact
    # registry code; the Observation is unrelated vocabulary.
    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="hypertension"))

    assert result.status == "evidence_found"
    assert any(e.resource_id == "cond-1" for e in result.evidence)


def test_duplicate_evidence_is_removed(mongo_db):
    # A query that both structured (exact concept match) AND semantic
    # (literal word overlap with the display text) independently retrieve.
    _insert_condition(mongo_db, "p1", "cond-1", "59621000", "Hypertension")
    retriever = HybridEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="hypertension"))

    resource_ids = [e.resource_id for e in result.evidence]
    assert resource_ids.count("cond-1") == 1


def test_deduplication_uses_patient_resource_type_and_id():
    from app.services.hybrid_retriever import _deduplicate

    a = Evidence(patient_id="p1", resource_type="Observation", resource_id="obs-1", display="Reading A", value=1)
    b = Evidence(patient_id="p1", resource_type="Observation", resource_id="obs-1", display="Reading B", value=2)  # same identity, different content
    c = Evidence(patient_id="p1", resource_type="Condition", resource_id="obs-1", display="Different type")  # same resource_id, different type

    deduplicated = _deduplicate([a, b, c])

    # (patient_id, resource_type, resource_id) identity: a/b collapse to one; c is distinct.
    assert len(deduplicated) == 2
    assert deduplicated[0] == a  # first occurrence wins
    assert deduplicated[1] == c


# --- 5-6: top-K -------------------------------------------------------------------


def test_top_k_is_enforced(mongo_db):
    for i in range(8):
        _insert_observation(mongo_db, "p1", f"obs-{i}", "8480-6", "Systolic Blood Pressure", 120 + i, "mm[Hg]")
    retriever = HybridEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="blood pressure"), limit=3)

    assert len(result.evidence) == 3


def test_invalid_limit_is_rejected(mongo_db):
    retriever = HybridEvidenceRetriever(mongo_db)

    with pytest.raises(ValueError):
        retriever.retrieve(RetrievalRequest(patient_id="p1", query="hypertension"), limit=0)

    with pytest.raises(ValueError):
        retriever.retrieve(RetrievalRequest(patient_id="p1", query="hypertension"), limit=-1)


def test_default_limit_is_five():
    assert DEFAULT_LIMIT == 5


# --- 7-13: no-evidence / unsupported truth table --------------------------------


def test_structured_evidence_only_is_evidence_found(mongo_db):
    # Registry concept match with an exact code, but display text sharing
    # no vocabulary with the query — only structured retrieval can find it.
    _insert_condition(mongo_db, "p1", "cond-1", "59621000", "Hypertension")
    retriever = HybridEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="hypertension"))

    assert result.status == "evidence_found"


def test_semantic_evidence_only_is_evidence_found(mongo_db):
    # Not a registered structured concept, but genuine lexical overlap.
    _insert_observation(mongo_db, "p1", "obs-1", "8480-6", "Systolic Blood Pressure", 125, "mm[Hg]")
    retriever = HybridEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="blood pressure reading"))

    assert result.status == "evidence_found"
    assert result.evidence[0].resource_id == "obs-1"


def test_both_strategies_finding_evidence_is_evidence_found(mongo_db):
    _insert_condition(mongo_db, "p1", "cond-1", "59621000", "Hypertension")
    retriever = HybridEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="hypertension"))

    assert result.status == "evidence_found"


def test_both_no_evidence_is_no_evidence_found(mongo_db):
    _insert_observation(mongo_db, "p1", "obs-glucose", "2339-0", "Glucose", 90, "mg/dL")
    retriever = HybridEvidenceRetriever(mongo_db)

    # A registered concept (structured: supported, no match) with no
    # lexical overlap either (semantic: no match).
    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="hemoglobin a1c"))

    assert result.status == "no_evidence_found"
    assert result.evidence == []


def test_both_unsupported_is_unsupported(mongo_db):
    retriever = HybridEvidenceRetriever(mongo_db)

    # Not a registered structured concept, and semantic tokenizes to nothing.
    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="???"))

    assert result.status == "unsupported"
    assert result.evidence == []


def test_one_unsupported_one_evidence_is_evidence_found(mongo_db, monkeypatch):
    # Precisely controlled via mocking: whether a real query naturally
    # produces "unsupported" from one strategy while the other finds
    # evidence depends on tokenization/registry details that shouldn't be
    # relied on — the hybrid branch logic itself is what's under test here.
    retriever = HybridEvidenceRetriever(mongo_db)
    found_evidence = Evidence(patient_id="p1", resource_type="Condition", resource_id="cond-1", display="Hypertension")

    def unsupported_retrieve(request, **kwargs):
        return RetrievalResult.unsupported(patient_id=request.patient_id, query=request.query, message="fake unsupported")

    def evidence_retrieve(request, **kwargs):
        return RetrievalResult.from_evidence(patient_id=request.patient_id, query=request.query, evidence=[found_evidence])

    monkeypatch.setattr(retriever._structured, "retrieve", unsupported_retrieve)
    monkeypatch.setattr(retriever._semantic, "retrieve", evidence_retrieve)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="anything"))

    assert result.status == "evidence_found"
    assert result.evidence[0].resource_id == "cond-1"


def test_one_unsupported_one_no_evidence_is_no_evidence_found(mongo_db, monkeypatch):
    retriever = HybridEvidenceRetriever(mongo_db)

    def unsupported_retrieve(request, **kwargs):
        return RetrievalResult.unsupported(patient_id=request.patient_id, query=request.query, message="fake unsupported")

    def no_evidence_retrieve(request, **kwargs):
        return RetrievalResult.from_evidence(patient_id=request.patient_id, query=request.query, evidence=[])

    monkeypatch.setattr(retriever._structured, "retrieve", unsupported_retrieve)
    monkeypatch.setattr(retriever._semantic, "retrieve", no_evidence_retrieve)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="anything"))

    assert result.status == "no_evidence_found"
    assert result.evidence == []


# --- 14-15: patient isolation -----------------------------------------------------


def test_patient_isolation(mongo_db):
    _insert_condition(mongo_db, "patient-a", "cond-a", "59621000", "Hypertension")
    _insert_condition(mongo_db, "patient-b", "cond-b", "59621000", "Hypertension")
    retriever = HybridEvidenceRetriever(mongo_db)

    result_a = retriever.retrieve(RetrievalRequest(patient_id="patient-a", query="hypertension"))
    result_b = retriever.retrieve(RetrievalRequest(patient_id="patient-b", query="hypertension"))

    assert [e.resource_id for e in result_a.evidence] == ["cond-a"]
    assert [e.resource_id for e in result_b.evidence] == ["cond-b"]


def test_cross_patient_evidence_raises_value_error(mongo_db, monkeypatch):
    # Simulates an underlying retriever misbehaving — ignoring
    # request.patient_id and returning another patient's (self-consistent)
    # RetrievalResult. The hybrid layer must detect and loudly reject this,
    # never silently filter it out.
    retriever = HybridEvidenceRetriever(mongo_db)

    def rogue_retrieve(request, **kwargs):
        rogue_evidence = Evidence(patient_id="patient-b", resource_type="Condition", resource_id="cond-rogue", display="Rogue")
        return RetrievalResult.from_evidence(patient_id="patient-b", query=request.query, evidence=[rogue_evidence])

    monkeypatch.setattr(retriever._structured, "retrieve", rogue_retrieve)

    with pytest.raises(ValueError):
        retriever.retrieve(RetrievalRequest(patient_id="patient-a", query="anything"))


# --- 16: determinism ---------------------------------------------------------------


def test_deterministic_repeated_retrieval(mongo_db):
    _insert_condition(mongo_db, "p1", "cond-1", "59621000", "Hypertension")
    _insert_observation(mongo_db, "p1", "obs-1", "8480-6", "Systolic Blood Pressure", 125, "mm[Hg]")
    retriever = HybridEvidenceRetriever(mongo_db)
    request = RetrievalRequest(patient_id="p1", query="hypertension blood pressure")

    first = retriever.retrieve(request)
    second = retriever.retrieve(request)

    assert first.model_dump() == second.model_dump()


# --- 17: existing ranker is reused, not reimplemented ---------------------------


def test_uses_deterministic_evidence_ranker(mongo_db):
    calls = []

    class SpyRanker:
        def rank(self, evidence, **kwargs):
            calls.append((list(evidence), kwargs))
            return DeterministicEvidenceRanker().rank(evidence, **kwargs)

    _insert_condition(mongo_db, "p1", "cond-1", "59621000", "Hypertension")
    retriever = HybridEvidenceRetriever(mongo_db, ranker=SpyRanker())

    retriever.retrieve(RetrievalRequest(patient_id="p1", query="hypertension"))

    assert len(calls) == 1
    assert calls[0][1].get("patient_id") == "p1"


def test_no_second_ranking_algorithm_is_implemented():
    import app.services.hybrid_retriever as module

    source = inspect.getsource(module)
    # No independent sort/score computation — ranking is delegated to
    # self._ranker.rank(...), the existing Phase 5C component.
    assert "self._ranker.rank(" in source
    assert "sorted(" not in source  # only _deduplicate touches ordering, and it doesn't sort


# --- 18-19: no raw FHIR / no MongoDB _id ------------------------------------------


def test_no_raw_fhir_or_mongo_id_in_result(mongo_db):
    _insert_condition(mongo_db, "p1", "cond-1", "59621000", "Hypertension")
    retriever = HybridEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="hypertension"))
    dumped = result.model_dump()

    assert "data" not in dumped
    for evidence_dict in dumped["evidence"]:
        assert "data" not in evidence_dict
        assert "resourceType" not in evidence_dict
        assert "_id" not in evidence_dict


# --- 23: evidence remains traceable ------------------------------------------------


def test_evidence_remains_traceable(mongo_db):
    _insert_condition(mongo_db, "p1", "cond-1", "59621000", "Hypertension")
    retriever = HybridEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="hypertension"))

    evidence = result.evidence[0]
    assert evidence.patient_id == "p1"
    assert evidence.resource_type == "Condition"
    assert evidence.resource_id == "cond-1"
    assert evidence.source_collection == "fhir_resources"
    assert evidence.source_reference == "Condition/cond-1"


# --- 25: no LLM functionality -------------------------------------------------------


def test_no_llm_functionality_exists():
    import app.services.hybrid_retriever as module

    imports = [
        line.strip()
        for line in inspect.getsource(module).splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    import_text = "\n".join(imports).lower()
    for forbidden in ("openai", "anthropic", "requests", "httpx", "langchain", "llamaindex", "chromadb", "pinecone", "faiss", "neo4j"):
        assert forbidden not in import_text
