"""The structured Evidence model: a lightweight, traceable reference to a
single relevant fact already stored in an ingested FHIR resource.

Evidence is NOT a new source of truth. The original FHIR JSON stored in
`fhir_resources` (see app/repositories/fhir_repository.py) remains the
source of truth — an Evidence object only points back to it and carries
the specific coding/value that was actually read from it. Nothing here is
inferred, generated, scored, or guessed: if a value did not come from a
real stored FHIR resource, it does not belong in this model.

A single reusable model covers every supported resource type (Condition,
MedicationRequest, AllergyIntolerance, Observation, Procedure, Encounter,
DiagnosticReport, and Patient demographics) rather than one model per
type — the fields that don't apply to a given resource type are simply
left unset.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from app.models.schemas import PaginationMeta


class Evidence(BaseModel):
    # --- traceability back to the source FHIR resource (required) ---
    patient_id: str = Field(..., min_length=1)
    resource_type: str = Field(..., min_length=1)
    resource_id: str = Field(..., min_length=1)
    source_collection: str = Field(default="fhir_resources", min_length=1)
    source_reference: Optional[str] = None

    # --- clinical content (optional — not every resource type/record carries every field) ---
    code: Optional[str] = None
    coding_system: Optional[str] = None
    display: Optional[str] = None
    value: Optional[Any] = None
    unit: Optional[str] = None
    effective_date: Optional[str] = None
    status: Optional[str] = None

    @model_validator(mode="after")
    def _default_source_reference(self) -> "Evidence":
        """If not explicitly supplied, source_reference is just a formatted
        view of the already-required resource_type/resource_id — never a
        generated or inferred value."""
        if not self.source_reference:
            self.source_reference = f"{self.resource_type}/{self.resource_id}"
        return self


class EvidenceListOut(BaseModel):
    items: list[Evidence]
    pagination: PaginationMeta
