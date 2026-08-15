"""The normalized PatientProfile: a flat, application-friendly view built
deterministically from the raw FHIR resources already stored by Phase 1.

Every clinical item keeps its source `resource_id` so it can always be
traced back to the original FHIR resource in `fhir_resources` — raw FHIR
remains the source of truth, this is a derived convenience representation.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class TelecomOut(BaseModel):
    system: Optional[str] = None
    value: Optional[str] = None
    use: Optional[str] = None


class DemographicsOut(BaseModel):
    patient_id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    full_name: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    race: Optional[str] = None
    ethnicity: Optional[str] = None
    marital_status: Optional[str] = None
    deceased: bool = False
    deceased_date: Optional[str] = None


class ContactInfoOut(BaseModel):
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    telecom: list[TelecomOut] = []


class ConditionOut(BaseModel):
    resource_id: str
    code: Optional[str] = None
    display: Optional[str] = None
    system: Optional[str] = None
    clinical_status: Optional[str] = None
    verification_status: Optional[str] = None
    onset_date: Optional[str] = None
    recorded_date: Optional[str] = None


class ObservationOut(BaseModel):
    resource_id: str
    code: Optional[str] = None
    name: Optional[str] = None
    system: Optional[str] = None
    value: Optional[Any] = None
    value_type: Optional[str] = None
    unit: Optional[str] = None
    reference_range: Optional[str] = None
    interpretation: Optional[str] = None
    effective_date: Optional[str] = None
    status: Optional[str] = None


class MedicationOut(BaseModel):
    resource_id: str
    medication_name: Optional[str] = None
    code: Optional[str] = None
    system: Optional[str] = None
    status: Optional[str] = None
    intent: Optional[str] = None
    dosage_text: Optional[str] = None
    route: Optional[str] = None
    frequency: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class ProcedureOut(BaseModel):
    resource_id: str
    code: Optional[str] = None
    name: Optional[str] = None
    system: Optional[str] = None
    status: Optional[str] = None
    performed_date: Optional[str] = None
    reason: Optional[str] = None


class EncounterOut(BaseModel):
    resource_id: str
    encounter_type: Optional[str] = None
    status: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    reason: Optional[str] = None
    location: Optional[str] = None


class DiagnosticReportOut(BaseModel):
    resource_id: str
    report_name: Optional[str] = None
    status: Optional[str] = None
    category: Optional[str] = None
    effective_date: Optional[str] = None
    result_references: list[str] = []


class AllergyReactionOut(BaseModel):
    manifestation: list[str] = []
    severity: Optional[str] = None


class AllergyOut(BaseModel):
    resource_id: str
    substance: Optional[str] = None
    code: Optional[str] = None
    system: Optional[str] = None
    clinical_status: Optional[str] = None
    verification_status: Optional[str] = None
    reaction: list[AllergyReactionOut] = []
    severity: Optional[str] = None
    manifestation: list[str] = []


class PatientProfileOut(BaseModel):
    patient_id: str
    demographics: DemographicsOut
    contact: ContactInfoOut
    conditions: list[ConditionOut] = []
    observations: list[ObservationOut] = []
    medications: list[MedicationOut] = []
    procedures: list[ProcedureOut] = []
    encounters: list[EncounterOut] = []
    diagnostic_reports: list[DiagnosticReportOut] = []
    allergies: list[AllergyOut] = []
    normalized_at: Optional[datetime] = None
