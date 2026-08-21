"""Regression test for a real bug: AskService.ask() assembled the final
prompt string from prompt.instructions + prompt.status_note +
prompt.evidence_text only — prompt.query (the user's actual question) was
never included anywhere in what was sent to the LLM. This went undetected
because every existing test either used a fake provider that ignores
prompt content entirely, or independently re-built the same
(equally incomplete) 3-field join rather than capturing what
AskService.ask() itself actually sends. Confirmed live against a real
provider: without the question, the model correctly (per its own
instructions to never guess) asked the user to clarify what they wanted to
know, instead of answering — which looked like a grounding/citation
failure but was actually a missing-question bug."""

from app.services.ask_service import AskService


def _insert_medication(mongo_db, patient_id, resource_id, code, display, status="active"):
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
                "authoredOn": "2023-01-01",
            },
        }
    )


class _SpyLLMProvider:
    def __init__(self, response: str = "Noted [MedicationRequest/med-1]."):
        self.prompts_received: list[str] = []
        self._response = response

    def generate(self, prompt: str) -> str:
        self.prompts_received.append(prompt)
        return self._response


def test_the_actual_question_reaches_the_llm_prompt(mongo_db):
    _insert_medication(mongo_db, "p1", "med-1", "313782", "Acetaminophen 325 MG Oral Tablet")
    provider = _SpyLLMProvider()
    query = "What medications is the patient taking?"

    AskService(mongo_db, provider).ask("p1", query)

    assert len(provider.prompts_received) == 1
    assert query in provider.prompts_received[0]


def test_a_different_question_also_reaches_the_prompt_correctly(mongo_db):
    _insert_medication(mongo_db, "p1", "med-1", "313782", "Acetaminophen 325 MG Oral Tablet")
    provider = _SpyLLMProvider()
    query = "What is the dosage status of the patient's medications?"

    AskService(mongo_db, provider).ask("p1", query)

    assert query in provider.prompts_received[0]


def test_question_appears_alongside_instructions_and_evidence_not_instead_of_them(mongo_db):
    _insert_medication(mongo_db, "p1", "med-1", "313782", "Acetaminophen 325 MG Oral Tablet")
    provider = _SpyLLMProvider()

    AskService(mongo_db, provider).ask("p1", "What medications is the patient taking?")

    sent = provider.prompts_received[0]
    assert "traceable to a specific evidence item" in sent  # instructions
    assert "[MedicationRequest/med-1]" in sent  # evidence
    assert "What medications is the patient taking?" in sent  # the actual question
