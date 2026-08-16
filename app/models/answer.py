"""GroundedAnswer: the CONTRACT for a future grounded AI answer — model
layer only. No LLM call and no answer generation exist yet (that's a later
Phase 6 sub-phase); this only defines the shape any future answer-producer
must fill in.

Mirrors the same safety pattern already established by RetrievalResult
(app/models/retrieval.py) and GroundedContext (app/models/rag_context.py):
a closed three-state status, and validators that make it structurally
impossible to claim an answer without evidence, or to attach evidence
belonging to a different patient.

NO EVIDENCE != NEGATIVE FACT, and NO ANSWER != A FABRICATED ONE: whenever
status is not "answered", both `answer_text` and `evidence` must be empty
— there is no way to construct a GroundedAnswer that looks answered while
actually resting on nothing.

Deliberately excluded, even though a future implementation might be
tempted to add them: confidence_score, hallucination_score,
medical_probability, ai_reasoning, chain_of_thought, or any other hidden
reasoning/arbitrary-metadata field. An answer is either grounded in real,
attached Evidence, or it isn't produced at all.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.models.evidence import Evidence

AnswerStatus = Literal["answered", "insufficient_evidence", "unsupported"]


class GroundedAnswer(BaseModel):
    """A future grounded answer, traceable to the Evidence that supports
    it. Contains no scoring, no reasoning trace, no arbitrary metadata —
    only what's needed to state an answer and cite exactly what backs it."""

    patient_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    status: AnswerStatus
    answer_text: Optional[str] = None
    evidence: list[Evidence] = []
    # Human-readable context for insufficient_evidence/unsupported — e.g.
    # why no answer could be given. Never a clinical fact, never reasoning.
    message: Optional[str] = None

    @model_validator(mode="after")
    def _validate_status_matches_answer_and_evidence(self) -> "GroundedAnswer":
        if self.status == "answered":
            if not self.answer_text:
                raise ValueError("status 'answered' requires non-empty answer_text")
            if not self.evidence:
                raise ValueError("status 'answered' requires at least one evidence item")
        else:
            if self.answer_text:
                raise ValueError(f"status '{self.status}' must not carry answer_text")
            if self.evidence:
                raise ValueError(f"status '{self.status}' must not carry evidence items")
        return self

    @model_validator(mode="after")
    def _validate_evidence_belongs_to_requested_patient(self) -> "GroundedAnswer":
        mismatched = [e for e in self.evidence if e.patient_id != self.patient_id]
        if mismatched:
            raise ValueError(
                f"GroundedAnswer for patient '{self.patient_id}' contains evidence belonging to a "
                f"different patient ('{mismatched[0].patient_id}') — cross-patient evidence is not permitted"
            )
        return self


class AskRequest(BaseModel):
    """The request body for POST /api/patients/{patient_id}/ask (Phase 6G).
    patient_id itself is always a URL path parameter, never taken from the
    body — see app/api/routes/patients.py."""

    query: str = Field(..., min_length=1)
