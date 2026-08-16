"""Phase 10C: patient isolation across the ENTIRE pipeline, specifically
for the harder case where Patient A and Patient B both have evidence
relevant to the SAME query. Both patients get their own "Blood Pressure"
Observation with different values/resource_ids, and both get asked the
same "blood pressure" query. This is a stronger test than asking about two
different conditions (which could pass isolation "by accident" merely
because the queries themselves are unrelated) — here the two patients'
evidence would plausibly compete for the same retrieval/ranking slot,
which is exactly the fragile spot cross-patient leakage bugs would hide.

Verifies: retrieval candidates, GroundedContext, the rendered prompt text,
the final GroundedAnswer, and the persisted AuditRecord — never contain the
other patient's evidence, in both directions. No real LLM is called."""

from app.models.retrieval import RetrievalRequest
from app.repositories import audit_repository
from app.services.ask_service import AskService
from app.services.grounded_prompt import build_grounded_prompt
from app.services.hybrid_retriever import HybridEvidenceRetriever
from app.services.rag_context import GroundedContextService


def _insert_blood_pressure(mongo_db, patient_id, resource_id, value):
    mongo_db["fhir_resources"].insert_one(
        {
            "patient_id": patient_id,
            "resource_type": "Observation",
            "resource_id": resource_id,
            "data": {
                "resourceType": "Observation",
                "id": resource_id,
                "status": "final",
                "code": {
                    "coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic Blood Pressure"}],
                    "text": "Systolic Blood Pressure",
                },
                "effectiveDateTime": "2024-01-01",
                "valueQuantity": {"value": value, "unit": "mm[Hg]"},
            },
        }
    )


class _FakeLLMProvider:
    """Cites whichever resource_id the prompt actually presented, proving
    the LLM only ever saw one patient's evidence at a time."""

    def __init__(self):
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        import re

        match = re.search(r"\[Observation/([\w-]+)\]", prompt)
        resource_id = match.group(1) if match else "unknown"
        return f"Blood pressure reading noted [Observation/{resource_id}]."


def _setup_two_patients_with_same_query_evidence(mongo_db):
    _insert_blood_pressure(mongo_db, "patient-a", "bp-a", value=118)
    _insert_blood_pressure(mongo_db, "patient-b", "bp-b", value=175)


# --- retrieval candidates never cross ------------------------------------------------------


def test_retrieval_candidates_never_cross_for_the_same_query(mongo_db):
    _setup_two_patients_with_same_query_evidence(mongo_db)
    retriever = HybridEvidenceRetriever(mongo_db)

    result_a = retriever.retrieve(RetrievalRequest(patient_id="patient-a", query="blood pressure"))
    result_b = retriever.retrieve(RetrievalRequest(patient_id="patient-b", query="blood pressure"))

    assert [e.resource_id for e in result_a.evidence] == ["bp-a"]
    assert [e.resource_id for e in result_b.evidence] == ["bp-b"]
    assert "bp-b" not in [e.resource_id for e in result_a.evidence]
    assert "bp-a" not in [e.resource_id for e in result_b.evidence]


# --- GroundedContext never crosses ----------------------------------------------------------


def test_grounded_context_never_crosses_for_the_same_query(mongo_db):
    _setup_two_patients_with_same_query_evidence(mongo_db)
    service = GroundedContextService(mongo_db)

    context_a = service.build_context(RetrievalRequest(patient_id="patient-a", query="blood pressure"))
    context_b = service.build_context(RetrievalRequest(patient_id="patient-b", query="blood pressure"))

    assert [e.resource_id for e in context_a.evidence] == ["bp-a"]
    assert [e.resource_id for e in context_b.evidence] == ["bp-b"]
    assert all(e.patient_id == "patient-a" for e in context_a.evidence)
    assert all(e.patient_id == "patient-b" for e in context_b.evidence)


# --- the rendered prompt text never crosses --------------------------------------------------


def test_prompt_text_never_contains_the_other_patients_evidence(mongo_db):
    _setup_two_patients_with_same_query_evidence(mongo_db)
    service = GroundedContextService(mongo_db)

    context_a = service.build_context(RetrievalRequest(patient_id="patient-a", query="blood pressure"))
    context_b = service.build_context(RetrievalRequest(patient_id="patient-b", query="blood pressure"))
    prompt_a = build_grounded_prompt(context_a)
    prompt_b = build_grounded_prompt(context_b)

    assert "bp-b" not in prompt_a.evidence_text
    assert "175" not in prompt_a.evidence_text
    assert "bp-a" not in prompt_b.evidence_text
    assert "118" not in prompt_b.evidence_text


# --- the final GroundedAnswer never crosses ---------------------------------------------------


def test_grounded_answer_never_crosses_for_the_same_query(mongo_db):
    _setup_two_patients_with_same_query_evidence(mongo_db)
    provider = _FakeLLMProvider()

    answer_a = AskService(mongo_db, provider).ask("patient-a", "blood pressure")
    answer_b = AskService(mongo_db, provider).ask("patient-b", "blood pressure")

    assert answer_a.status == "answered"
    assert [e.resource_id for e in answer_a.evidence] == ["bp-a"]
    assert answer_b.status == "answered"
    assert [e.resource_id for e in answer_b.evidence] == ["bp-b"]

    # Every prompt the (shared) fake provider received contained only one
    # patient's resource_id, never both.
    assert len(provider.calls) == 2
    assert "bp-b" not in provider.calls[0] and "bp-a" not in provider.calls[1]


# --- the persisted AuditRecord never crosses ---------------------------------------------------


def test_audit_record_never_crosses_for_the_same_query(mongo_db):
    _setup_two_patients_with_same_query_evidence(mongo_db)
    provider = _FakeLLMProvider()

    AskService(mongo_db, provider).ask("patient-a", "blood pressure")
    AskService(mongo_db, provider).ask("patient-b", "blood pressure")

    records_a, total_a = audit_repository.get_patient_audit_history(mongo_db, "patient-a")
    records_b, total_b = audit_repository.get_patient_audit_history(mongo_db, "patient-b")

    assert total_a == 1 and total_b == 1
    assert [r.resource_id for r in records_a[0].evidence_references] == ["bp-a"]
    assert [r.resource_id for r in records_b[0].evidence_references] == ["bp-b"]
    assert records_a[0].patient_id == "patient-a"
    assert records_b[0].patient_id == "patient-b"


# --- reverse-order check: asking B first, then A, still isolates correctly -------------------


def test_isolation_holds_regardless_of_request_order(mongo_db):
    _setup_two_patients_with_same_query_evidence(mongo_db)
    provider = _FakeLLMProvider()

    answer_b_first = AskService(mongo_db, provider).ask("patient-b", "blood pressure")
    answer_a_second = AskService(mongo_db, provider).ask("patient-a", "blood pressure")

    assert [e.resource_id for e in answer_b_first.evidence] == ["bp-b"]
    assert [e.resource_id for e in answer_a_second.evidence] == ["bp-a"]
