"""Phase 5B: the first real EvidenceRetriever implementation —
deterministic structured retrieval. Uses mongomock for patient-scoped
unit/integration tests; a separate live read-only script (not part of
this automated suite) verifies behavior against the real dataset."""

import inspect

from app.models.retrieval import RetrievalRequest
from app.services.structured_retriever import StructuredEvidenceRetriever


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


# --- 1-2: supported query, with and without matching evidence ----------------


def test_supported_hba1c_query_retrieves_real_evidence(mongo_db):
    _insert_observation(mongo_db, "p1", "obs-hba1c", "4548-4", "Hemoglobin A1c", 5.81, "%")
    retriever = StructuredEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="What is the patient's HbA1c?"))

    assert result.status == "evidence_found"
    assert len(result.evidence) == 1
    assert result.evidence[0].value == 5.81
    assert result.evidence[0].code == "4548-4"


def test_supported_query_with_no_matching_evidence(mongo_db):
    # Patient exists (has other resources) but no HbA1c observation at all.
    _insert_observation(mongo_db, "p1", "obs-glucose", "2339-0", "Glucose", 90, "mg/dL")
    retriever = StructuredEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="What is the patient's HbA1c?"))

    assert result.status == "no_evidence_found"
    assert result.evidence == []


# --- 3-4: unsupported query never guesses --------------------------------------


def test_unsupported_query_returns_unsupported_status(mongo_db):
    retriever = StructuredEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="What is the patient's favorite color?"))

    assert result.status == "unsupported"
    assert result.evidence == []
    assert result.message is not None


def test_unsupported_query_never_guesses_a_concept(mongo_db):
    # Even with real, matching data in the DB, a query with no recognized
    # trigger phrase must never accidentally retrieve unrelated evidence.
    _insert_observation(mongo_db, "p1", "obs-hba1c", "4548-4", "Hemoglobin A1c", 5.81, "%")
    retriever = StructuredEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="Tell me a joke"))

    assert result.status == "unsupported"
    assert result.evidence == []


# --- 5: patient isolation -------------------------------------------------------


def test_patient_isolation(mongo_db):
    _insert_observation(mongo_db, "patient-a", "obs-a", "4548-4", "Hemoglobin A1c", 5.81, "%")
    _insert_observation(mongo_db, "patient-b", "obs-b", "4548-4", "Hemoglobin A1c", 9.2, "%")
    retriever = StructuredEvidenceRetriever(mongo_db)

    result_a = retriever.retrieve(RetrievalRequest(patient_id="patient-a", query="HbA1c?"))
    result_b = retriever.retrieve(RetrievalRequest(patient_id="patient-b", query="HbA1c?"))

    assert result_a.evidence[0].resource_id == "obs-a"
    assert result_a.evidence[0].value == 5.81
    assert result_b.evidence[0].resource_id == "obs-b"
    assert result_b.evidence[0].value == 9.2
    assert all(e.patient_id == "patient-a" for e in result_a.evidence)
    assert all(e.patient_id == "patient-b" for e in result_b.evidence)


def test_patient_a_never_receives_patient_b_evidence_when_a_has_none(mongo_db):
    _insert_observation(mongo_db, "patient-b", "obs-b", "4548-4", "Hemoglobin A1c", 9.2, "%")
    retriever = StructuredEvidenceRetriever(mongo_db)

    result_a = retriever.retrieve(RetrievalRequest(patient_id="patient-a", query="HbA1c?"))

    assert result_a.status == "no_evidence_found"
    assert result_a.evidence == []


# --- 6-7: traceability / no raw FHIR --------------------------------------------


def test_evidence_objects_remain_traceable(mongo_db):
    _insert_condition(mongo_db, "p1", "cond-htn", "59621000", "Hypertension")
    retriever = StructuredEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="Does the patient have hypertension?"))

    evidence = result.evidence[0]
    assert evidence.patient_id == "p1"
    assert evidence.resource_type == "Condition"
    assert evidence.resource_id == "cond-htn"
    assert evidence.source_collection == "fhir_resources"
    assert evidence.source_reference == "Condition/cond-htn"


def test_no_raw_fhir_document_in_result(mongo_db):
    _insert_condition(mongo_db, "p1", "cond-htn", "59621000", "Hypertension")
    retriever = StructuredEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="hypertension"))
    dumped = result.model_dump()

    assert "data" not in dumped
    for evidence_dict in dumped["evidence"]:
        assert "data" not in evidence_dict
        assert "resourceType" not in evidence_dict
        assert "_id" not in evidence_dict


# --- 8: determinism ---------------------------------------------------------------


def test_deterministic_repeated_retrieval(mongo_db):
    _insert_observation(mongo_db, "p1", "obs-hba1c", "4548-4", "Hemoglobin A1c", 5.81, "%")
    retriever = StructuredEvidenceRetriever(mongo_db)
    request = RetrievalRequest(patient_id="p1", query="What is the patient's HbA1c?")

    first = retriever.retrieve(request)
    second = retriever.retrieve(request)

    assert first.model_dump() == second.model_dump()


# --- 9: no evidence is never converted into a negative claim -------------------


def test_no_evidence_is_not_a_negative_clinical_claim(mongo_db):
    retriever = StructuredEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="Does the patient have hypertension?"))

    assert result.status == "no_evidence_found"
    # The status is a neutral retrieval state, not an assertion the patient
    # lacks the condition — nothing in the result claims a negative fact.
    if result.message:
        assert "does not have" not in result.message.lower()
        assert "no hypertension" not in result.message.lower()


# --- 10-11: patient-scoped queries, no unbounded/cross-patient query -----------


def test_retrieval_uses_patient_scoped_queries(mongo_db, monkeypatch):
    _insert_observation(mongo_db, "p1", "obs-hba1c", "4548-4", "Hemoglobin A1c", 5.81, "%")
    collection = mongo_db["fhir_resources"]
    original_find = collection.find
    captured = []

    def spying_find(filter=None, *args, **kwargs):
        captured.append(filter or {})
        return original_find(filter, *args, **kwargs)

    monkeypatch.setattr(collection, "find", spying_find)

    StructuredEvidenceRetriever(mongo_db).retrieve(RetrievalRequest(patient_id="p1", query="HbA1c?"))

    assert captured
    for query in captured:
        assert query != {}
        assert query.get("patient_id") == "p1"


def test_no_unbounded_query_pattern_in_module_source():
    import app.services.structured_retriever as module

    source = inspect.getsource(module)
    assert "find({})" not in source
    assert "list(collection.find" not in source
    assert "list(db[" not in source


# --- 12: existing Evidence objects preserved unmodified ------------------------


def test_existing_evidence_objects_preserved(mongo_db):
    _insert_observation(mongo_db, "p1", "obs-hba1c", "4548-4", "Hemoglobin A1c", 5.81, "%", effective_date="2023-06-01")
    from app.services.evidence_service import EvidenceService

    direct = list(EvidenceService(mongo_db).get_patient_evidence_by_code("p1", "4548-4", resource_type="Observation"))
    result = StructuredEvidenceRetriever(mongo_db).retrieve(RetrievalRequest(patient_id="p1", query="HbA1c?"))

    assert result.evidence == direct


# --- 13: satisfies the Phase 5A EvidenceRetriever protocol ----------------------


def test_satisfies_evidence_retriever_protocol(mongo_db):
    from app.models.retrieval import RetrievalResult

    retriever = StructuredEvidenceRetriever(mongo_db)
    assert hasattr(retriever, "retrieve")

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="HbA1c?"))
    assert isinstance(result, RetrievalResult)
