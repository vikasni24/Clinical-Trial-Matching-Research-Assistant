"""GroundedContext: the minimal package a future Phase 6 (LLM) consumer
will read from — never something an LLM writes to.

This is intentionally near-identical in shape to RetrievalResult
(app/models/retrieval.py) and carries the exact same safety guarantees,
because it's a pure, deterministic repackaging of one — see
app/services/rag_context.py. It contains Evidence objects only: no raw
FHIR document, no MongoDB _id, no generated clinical facts, no LLM
reasoning, no confidence/probability score of any kind.
"""

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from app.models.evidence import Evidence
from app.models.retrieval import RetrievalStatus


class GroundedContext(BaseModel):
    """query + status + Evidence, traceable to patient_id/resource_type/
    resource_id on every item. Preserves "no_evidence_found" and
    "unsupported" explicitly — neither is ever silently upgraded to a
    success response with fabricated content."""

    patient_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    status: RetrievalStatus
    evidence: list[Evidence] = []
    message: Optional[str] = None

    @model_validator(mode="after")
    def _validate_status_matches_evidence(self) -> "GroundedContext":
        if self.status == "evidence_found" and not self.evidence:
            raise ValueError("status 'evidence_found' requires at least one evidence item")
        if self.status != "evidence_found" and self.evidence:
            raise ValueError(f"status '{self.status}' must not carry evidence items")
        return self

    @model_validator(mode="after")
    def _validate_evidence_belongs_to_requested_patient(self) -> "GroundedContext":
        mismatched = [e for e in self.evidence if e.patient_id != self.patient_id]
        if mismatched:
            raise ValueError(
                f"GroundedContext for patient '{self.patient_id}' contains evidence belonging to a "
                f"different patient ('{mismatched[0].patient_id}') — cross-patient evidence is not permitted"
            )
        return self
