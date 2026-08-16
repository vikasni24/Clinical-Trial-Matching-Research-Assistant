"""The retrieval CONTRACT for a future Hybrid RAG system — request/result
data shapes only. No retrieval strategy is implemented here (see
app/services/retrieval.py for the provider-independent interface those
future strategies — deterministic structured, keyword, vector, graph,
hybrid — will implement).

This contract composes the existing Evidence model (app/models/evidence.py)
unchanged; it never copies or reinvents clinical facts. Its entire purpose
is to make it structurally impossible for downstream RAG/LLM code to
mistake "nothing was found" or "this couldn't be retrieved" for an
established patient fact:

  - RetrievalResult.status is a closed three-state value — "evidence_found"
    only when there IS evidence; "no_evidence_found" / "unsupported" only
    when there is NONE. Constructing a result with mismatched status and
    evidence raises a validation error rather than silently allowing it.
  - Every Evidence item's patient_id is validated to match the result's own
    patient_id — a RetrievalResult literally cannot be constructed with
    another patient's evidence attached.

NO EVIDENCE != NEGATIVE FACT: an empty/unsupported result never implies the
patient lacks the queried condition/value — it only means retrieval found
nothing, which is a different, explicit state.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from app.models.evidence import Evidence

RetrievalStatus = Literal["evidence_found", "no_evidence_found", "unsupported"]


class RetrievalRequest(BaseModel):
    """A patient-scoped request for evidence. patient_id is mandatory and
    is the sole scoping mechanism a retrieval implementation may use —
    there is no way to express an unscoped ("all patients") request."""

    patient_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)


class RetrievalResult(BaseModel):
    """What a retrieval strategy returns. Always echoes the requested
    patient_id and query for traceability; never carries evidence for a
    different patient (enforced below)."""

    patient_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    status: RetrievalStatus
    evidence: list[Evidence] = []
    # Human-readable context — especially for "unsupported" (e.g. why no
    # strategy could service this query yet). Never a clinical fact.
    message: Optional[str] = None

    @model_validator(mode="after")
    def _validate_status_matches_evidence(self) -> "RetrievalResult":
        if self.status == "evidence_found" and not self.evidence:
            raise ValueError("status 'evidence_found' requires at least one evidence item")
        if self.status != "evidence_found" and self.evidence:
            raise ValueError(f"status '{self.status}' must not carry evidence items")
        return self

    @model_validator(mode="after")
    def _validate_evidence_belongs_to_requested_patient(self) -> "RetrievalResult":
        mismatched = [e for e in self.evidence if e.patient_id != self.patient_id]
        if mismatched:
            raise ValueError(
                f"RetrievalResult for patient '{self.patient_id}' contains evidence belonging to a "
                f"different patient ('{mismatched[0].patient_id}') — cross-patient evidence is not permitted"
            )
        return self

    @classmethod
    def from_evidence(cls, patient_id: str, query: str, evidence: list[Evidence]) -> "RetrievalResult":
        """Deterministically wraps already-retrieved Evidence into a valid
        RetrievalResult. Performs no retrieval itself — the caller must
        already have fetched `evidence` (e.g. via EvidenceService). status
        is derived purely from whether anything was found; nothing here
        is guessed, scored, or generated."""
        if evidence:
            return cls(patient_id=patient_id, query=query, status="evidence_found", evidence=evidence)
        return cls(patient_id=patient_id, query=query, status="no_evidence_found", evidence=[])

    @classmethod
    def unsupported(cls, patient_id: str, query: str, message: str) -> "RetrievalResult":
        """A request that could not be serviced at all (e.g. no retrieval
        strategy exists yet for this kind of query) — distinct from a
        supported retrieval that simply found nothing."""
        return cls(patient_id=patient_id, query=query, status="unsupported", evidence=[], message=message)
