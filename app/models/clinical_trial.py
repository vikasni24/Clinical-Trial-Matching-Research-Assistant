"""The normalized ClinicalTrial representation.

Eligibility is split into two parts, mirroring how the matcher consumes it:
  - top-level demographic fields (minimum_age/maximum_age/sex) evaluated
    against PatientProfile.demographics
  - structured inclusion/exclusion criteria lists, each a small typed
    condition/medication/allergy/observation-threshold check evaluated
    against the rest of the PatientProfile

Criteria are always structured data, never free-text — this phase does not
parse natural-language eligibility text.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.models.schemas import PaginationMeta

SYNTHETIC_DATA_DISCLAIMER = (
    "SYNTHETIC/DEVELOPMENT DATA ONLY. These clinical trial records are fictional fixtures "
    "used for local development and automated testing. They are not real clinical trials "
    "and are not registered with any regulatory body or trial registry."
)


class TrialConditionOut(BaseModel):
    name: str
    code: Optional[str] = None
    system: Optional[str] = None


class TrialInterventionOut(BaseModel):
    name: str
    type: Optional[str] = None
    description: Optional[str] = None


class TrialLocationOut(BaseModel):
    facility: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    recruiting_status: Optional[str] = None


class EligibilityCriterionOut(BaseModel):
    """One structured, machine-evaluable eligibility check.

    type: "condition" | "medication" | "allergy" | "observation_threshold"
    Placement in `inclusion_criteria` vs `exclusion_criteria` determines
    directionality — the same "condition_met" evaluation is reused for
    both; only the PASS/FAIL mapping differs (see EligibilityMatcher).
    """

    type: str
    label: str
    code: Optional[str] = None
    system: Optional[str] = None
    display: Optional[str] = None
    operator: Optional[str] = None  # observation_threshold only: ">=", "<=", ">", "<", "=="
    value: Optional[float] = None  # observation_threshold only
    unit: Optional[str] = None  # observation_threshold only


class TrialEligibilityOut(BaseModel):
    minimum_age: Optional[int] = None
    maximum_age: Optional[int] = None
    sex: Optional[str] = None  # "male" | "female" | "all" | None (unspecified)
    inclusion_criteria: list[EligibilityCriterionOut] = []
    exclusion_criteria: list[EligibilityCriterionOut] = []


class ClinicalTrialOut(BaseModel):
    trial_id: str
    title: Optional[str] = None
    brief_title: Optional[str] = None
    official_title: Optional[str] = None
    sponsor: Optional[str] = None
    study_type: Optional[str] = None
    phase: Optional[str] = None
    status: Optional[str] = None
    conditions: list[TrialConditionOut] = []
    interventions: list[TrialInterventionOut] = []
    eligibility: TrialEligibilityOut
    locations: list[TrialLocationOut] = []
    source: Optional[str] = None
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    last_updated: Optional[str] = None
    ingested_at: Optional[datetime] = None


class TrialListOut(BaseModel):
    items: list[ClinicalTrialOut]
    pagination: PaginationMeta
    disclaimer: str = SYNTHETIC_DATA_DISCLAIMER


class TrialIngestionReportOut(BaseModel):
    disclaimer: str = SYNTHETIC_DATA_DISCLAIMER
    trials_discovered: int
    trials_inserted: int
    trials_updated: int
    trials_failed: int
    errors: list[str] = []
