"""Phase 5E: GroundedContext and the service that builds it from a
retrieval result. Pure data-transformation tests — no LLM anywhere."""

import inspect

import pytest
from pydantic import ValidationError

from app.models.evidence import Evidence
from app.models.rag_context import GroundedContext
from app.models.retrieval import RetrievalRequest, RetrievalResult
from app.services.rag_context import GroundedContextService, build_grounded_context


def _evidence(resource_id="obs-1", patient_id="p1"):
    return Evidence(
        patient_id=patient_id,
        resource_type="Observation",
        resource_id=resource_id,
        code="8480-6",
        display="Systolic Blood Pressure",
        value=125,
        unit="mm[Hg]",
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


# --- 20: grounded context contains only Evidence ----------------------------------


def test_grounded_context_contains_only_evidence():
    result = RetrievalResult.from_evidence(patient_id="p1", query="blood pressure", evidence=[_evidence()])

    context = build_grounded_context(result)

    assert isinstance(context, GroundedContext)
    assert all(isinstance(e, Evidence) for e in context.evidence)
    assert context.evidence[0] == _evidence()


def test_grounded_context_has_no_raw_fhir_or_mongo_id():
    result = RetrievalResult.from_evidence(patient_id="p1", query="blood pressure", evidence=[_evidence()])

    context = build_grounded_context(result)
    dumped = context.model_dump()

    assert "data" not in dumped
    for evidence_dict in dumped["evidence"]:
        assert "data" not in evidence_dict
        assert "resourceType" not in evidence_dict
        assert "_id" not in evidence_dict


def test_grounded_context_has_no_confidence_or_llm_fields():
    prohibited = {"confidence_score", "clinical_probability", "hallucination_score", "ai_reasoning", "answer", "generated_text"}
    assert prohibited.isdisjoint(GroundedContext.model_fields.keys())


# --- 21-22: preserves no_evidence_found / unsupported ------------------------------


def test_grounded_context_preserves_no_evidence_found():
    result = RetrievalResult.from_evidence(patient_id="p1", query="hemoglobin a1c", evidence=[])

    context = build_grounded_context(result)

    assert context.status == "no_evidence_found"
    assert context.evidence == []


def test_grounded_context_preserves_unsupported():
    result = RetrievalResult.unsupported(patient_id="p1", query="???", message="No supported concept identified")

    context = build_grounded_context(result)

    assert context.status == "unsupported"
    assert context.evidence == []
    assert context.message == "No supported concept identified"


def test_neither_state_is_upgraded_to_a_fabricated_success():
    empty_result = RetrievalResult.from_evidence(patient_id="p1", query="q", evidence=[])
    unsupported_result = RetrievalResult.unsupported(patient_id="p1", query="q", message="m")

    assert build_grounded_context(empty_result).status != "evidence_found"
    assert build_grounded_context(unsupported_result).status != "evidence_found"


# --- consistency / isolation validators mirror RetrievalResult --------------------


def test_grounded_context_rejects_status_evidence_mismatch():
    with pytest.raises(ValidationError):
        GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[])


def test_grounded_context_rejects_cross_patient_evidence():
    other_patient_evidence = _evidence(patient_id="p2")

    with pytest.raises(ValidationError):
        GroundedContext(patient_id="p1", query="q", status="evidence_found", evidence=[other_patient_evidence])


# --- build_grounded_context is a pure, deterministic mapping ----------------------


def test_build_grounded_context_is_a_pure_field_mapping():
    result = RetrievalResult.from_evidence(patient_id="p1", query="blood pressure", evidence=[_evidence()])

    context = build_grounded_context(result)

    assert context.patient_id == result.patient_id
    assert context.query == result.query
    assert context.status == result.status
    assert context.evidence == result.evidence
    assert context.message == result.message


def test_no_llm_functionality_in_rag_context_module():
    import app.services.rag_context as module

    imports = [
        line.strip()
        for line in inspect.getsource(module).splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    import_text = "\n".join(imports).lower()
    for forbidden in ("openai", "anthropic", "requests", "httpx", "langchain", "llamaindex"):
        assert forbidden not in import_text


# --- GroundedContextService: query -> Hybrid retrieval -> GroundedContext --------


def test_grounded_context_service_chains_hybrid_retrieval(mongo_db):
    _insert_condition(mongo_db, "p1", "cond-1", "59621000", "Hypertension")
    service = GroundedContextService(mongo_db)

    context = service.build_context(RetrievalRequest(patient_id="p1", query="hypertension"))

    assert context.status == "evidence_found"
    assert context.evidence[0].resource_id == "cond-1"


def test_grounded_context_service_respects_limit(mongo_db):
    for i in range(8):
        mongo_db["fhir_resources"].insert_one(
            {
                "patient_id": "p1",
                "resource_type": "Observation",
                "resource_id": f"obs-{i}",
                "data": {
                    "resourceType": "Observation",
                    "id": f"obs-{i}",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic Blood Pressure"}]},
                    "effectiveDateTime": f"20{10 + i:02d}-01-01",
                    "valueQuantity": {"value": 120 + i, "unit": "mm[Hg]"},
                },
            }
        )
    service = GroundedContextService(mongo_db)

    context = service.build_context(RetrievalRequest(patient_id="p1", query="blood pressure"), limit=2)

    assert len(context.evidence) == 2
