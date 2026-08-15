"""Deterministic normalization of raw FHIR resources (already ingested into
MongoDB by the Phase 1 pipeline) into a PatientProfile: a flat,
application-friendly representation for downstream clinical-trial matching.

Normalization is pure and deterministic: no LLMs, no embeddings, no
inference beyond what is explicitly encoded in the source FHIR. A field is
left None/empty whenever the underlying FHIR data does not carry it.

Run locally with:
    python -m app.services.patient_normalization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from pymongo.database import Database

from app.db.mongodb import ensure_indexes, get_database
from app.models.patient_profile import (
    AllergyOut,
    AllergyReactionOut,
    ConditionOut,
    ContactInfoOut,
    DemographicsOut,
    DiagnosticReportOut,
    EncounterOut,
    MedicationOut,
    ObservationOut,
    PatientProfileOut,
    ProcedureOut,
    TelecomOut,
)
from app.repositories import fhir_repository, patient_profile_repository
from app.services.fhir_parser import strip_reference

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic CodeableConcept / Coding helpers, shared by every normalizer below.
# ---------------------------------------------------------------------------


def _first_coding(codeable_concept: Optional[dict]) -> dict:
    if not isinstance(codeable_concept, dict):
        return {}
    codings = codeable_concept.get("coding")
    if isinstance(codings, list) and codings and isinstance(codings[0], dict):
        return codings[0]
    return {}


def _coded_text(codeable_concept: Optional[dict], fallback_to_code: bool = False) -> Optional[str]:
    """Best human-readable label for a CodeableConcept: `.text`, else the
    primary coding's `.display`, else (optionally) its bare `.code` — status
    codes like "active"/"confirmed" carry no separate display in Synthea but
    are already human-readable, so callers that know this set
    fallback_to_code=True."""
    if not isinstance(codeable_concept, dict):
        return None
    text = codeable_concept.get("text")
    if text:
        return text
    coding = _first_coding(codeable_concept)
    if coding.get("display"):
        return coding["display"]
    if fallback_to_code:
        return coding.get("code")
    return None


def _extension_text(resource: dict[str, Any], extension_url: str) -> Optional[str]:
    """US Core race/ethnicity extensions nest a human-readable label under a
    "text" sub-extension, with an ombCategory valueCoding as a fallback."""
    for ext in resource.get("extension") or []:
        if not isinstance(ext, dict) or ext.get("url") != extension_url:
            continue
        sub_extensions = ext.get("extension") or []
        for sub in sub_extensions:
            if isinstance(sub, dict) and sub.get("url") == "text" and sub.get("valueString"):
                return sub["valueString"]
        for sub in sub_extensions:
            if isinstance(sub, dict) and sub.get("url") == "ombCategory":
                coding = sub.get("valueCoding")
                if isinstance(coding, dict) and coding.get("display"):
                    return coding["display"]
        return None
    return None


def _reference_range_text(range_entry: dict[str, Any]) -> Optional[str]:
    if range_entry.get("text"):
        return range_entry["text"]

    low = range_entry.get("low") if isinstance(range_entry.get("low"), dict) else {}
    high = range_entry.get("high") if isinstance(range_entry.get("high"), dict) else {}
    low_value, high_value = low.get("value"), high.get("value")
    unit = low.get("unit") or high.get("unit") or ""

    if low_value is not None and high_value is not None:
        return f"{low_value}-{high_value}" + (f" {unit}" if unit else "")
    if low_value is not None:
        return f">= {low_value}" + (f" {unit}" if unit else "")
    if high_value is not None:
        return f"<= {high_value}" + (f" {unit}" if unit else "")
    return None


_VALUE_FIELDS = (
    ("valueQuantity", "Quantity"),
    ("valueCodeableConcept", "CodeableConcept"),
    ("valueString", "string"),
    ("valueBoolean", "boolean"),
    ("valueInteger", "integer"),
    ("valueRange", "Range"),
    ("valueRatio", "Ratio"),
    ("valueDateTime", "dateTime"),
    ("valuePeriod", "Period"),
)


def _extract_value(data: dict[str, Any]) -> tuple[Optional[Any], Optional[str], Optional[str]]:
    """Returns (value, value_type, unit) from whichever FHIR value[x] field
    is present on an Observation."""
    for field_name, type_name in _VALUE_FIELDS:
        if field_name not in data:
            continue
        raw_value = data[field_name]
        if field_name == "valueQuantity" and isinstance(raw_value, dict):
            return raw_value.get("value"), type_name, raw_value.get("unit")
        if field_name == "valueCodeableConcept":
            return _coded_text(raw_value, fallback_to_code=True), type_name, None
        return raw_value, type_name, None
    return None, None, None


# ---------------------------------------------------------------------------
# Patient-level normalization.
# ---------------------------------------------------------------------------


def normalize_demographics(patient_data: dict[str, Any]) -> DemographicsOut:
    names = [n for n in (patient_data.get("name") or []) if isinstance(n, dict)]
    primary_name = names[0] if names else {}
    given = primary_name.get("given") or []
    first_name = given[0] if given else None
    last_name = primary_name.get("family")
    full_name_parts = [part for part in [*given, last_name] if part]
    full_name = " ".join(full_name_parts) if full_name_parts else None

    deceased_date = patient_data.get("deceasedDateTime")
    deceased = bool(patient_data.get("deceasedBoolean")) or bool(deceased_date)

    return DemographicsOut(
        patient_id=patient_data.get("id"),
        first_name=first_name,
        last_name=last_name,
        full_name=full_name,
        date_of_birth=patient_data.get("birthDate"),
        gender=patient_data.get("gender"),
        race=_extension_text(patient_data, "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race"),
        ethnicity=_extension_text(
            patient_data, "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity"
        ),
        marital_status=_coded_text(patient_data.get("maritalStatus"), fallback_to_code=True),
        deceased=deceased,
        deceased_date=deceased_date,
    )


def normalize_contact(patient_data: dict[str, Any]) -> ContactInfoOut:
    addresses = [a for a in (patient_data.get("address") or []) if isinstance(a, dict)]
    primary_address = addresses[0] if addresses else {}
    lines = primary_address.get("line") or []

    telecoms = [
        TelecomOut(system=t.get("system"), value=t.get("value"), use=t.get("use"))
        for t in (patient_data.get("telecom") or [])
        if isinstance(t, dict)
    ]

    return ContactInfoOut(
        address=", ".join(lines) if lines else None,
        city=primary_address.get("city"),
        state=primary_address.get("state"),
        postal_code=primary_address.get("postalCode"),
        country=primary_address.get("country"),
        telecom=telecoms,
    )


# ---------------------------------------------------------------------------
# Per-resource-type normalization. Each takes a stored fhir_resources
# document ({resource_id, patient_id, data, ...}) and returns one item.
# ---------------------------------------------------------------------------


def normalize_condition(doc: dict[str, Any]) -> ConditionOut:
    data = doc["data"]
    code_field = data.get("code") or {}
    coding = _first_coding(code_field)

    return ConditionOut(
        resource_id=doc["resource_id"],
        code=coding.get("code"),
        display=code_field.get("text") or coding.get("display"),
        system=coding.get("system"),
        clinical_status=_coded_text(data.get("clinicalStatus"), fallback_to_code=True),
        verification_status=_coded_text(data.get("verificationStatus"), fallback_to_code=True),
        onset_date=data.get("onsetDateTime"),
        recorded_date=data.get("recordedDate"),
    )


def normalize_observation(doc: dict[str, Any]) -> ObservationOut:
    data = doc["data"]
    code_field = data.get("code") or {}
    coding = _first_coding(code_field)
    value, value_type, unit = _extract_value(data)

    reference_range = None
    ranges = data.get("referenceRange")
    if isinstance(ranges, list) and ranges and isinstance(ranges[0], dict):
        reference_range = _reference_range_text(ranges[0])

    interpretation = None
    interpretations = data.get("interpretation")
    if isinstance(interpretations, list) and interpretations:
        interpretation = _coded_text(interpretations[0], fallback_to_code=True)

    return ObservationOut(
        resource_id=doc["resource_id"],
        code=coding.get("code"),
        name=code_field.get("text") or coding.get("display"),
        system=coding.get("system"),
        value=value,
        value_type=value_type,
        unit=unit,
        reference_range=reference_range,
        interpretation=interpretation,
        effective_date=data.get("effectiveDateTime"),
        status=data.get("status"),
    )


def normalize_medication(doc: dict[str, Any]) -> MedicationOut:
    data = doc["data"]
    med_concept = data.get("medicationCodeableConcept") or {}
    coding = _first_coding(med_concept)

    dosage_list = data.get("dosageInstruction") or []
    dosage = dosage_list[0] if dosage_list and isinstance(dosage_list[0], dict) else {}

    frequency = None
    repeat = dosage.get("timing", {}).get("repeat") if isinstance(dosage.get("timing"), dict) else None
    if isinstance(repeat, dict) and repeat.get("frequency") and repeat.get("period"):
        period_unit = repeat.get("periodUnit", "")
        frequency = f"{repeat['frequency']} time(s) per {repeat['period']}{period_unit}"

    end_date = None
    validity_period = data.get("dispenseRequest", {}).get("validityPeriod")
    if isinstance(validity_period, dict):
        end_date = validity_period.get("end")

    return MedicationOut(
        resource_id=doc["resource_id"],
        medication_name=med_concept.get("text") or coding.get("display"),
        code=coding.get("code"),
        system=coding.get("system"),
        status=data.get("status"),
        intent=data.get("intent"),
        dosage_text=dosage.get("text"),
        route=_coded_text(dosage.get("route"), fallback_to_code=True),
        frequency=frequency,
        start_date=data.get("authoredOn"),
        end_date=end_date,
    )


def normalize_procedure(doc: dict[str, Any]) -> ProcedureOut:
    data = doc["data"]
    code_field = data.get("code") or {}
    coding = _first_coding(code_field)

    performed_date = data.get("performedDateTime")
    if not performed_date and isinstance(data.get("performedPeriod"), dict):
        performed_date = data["performedPeriod"].get("start")

    reason = None
    reason_codes = data.get("reasonCode")
    if isinstance(reason_codes, list) and reason_codes:
        reason = _coded_text(reason_codes[0], fallback_to_code=True)
    if reason is None:
        reason_refs = data.get("reasonReference")
        if isinstance(reason_refs, list) and reason_refs and isinstance(reason_refs[0], dict):
            reason = reason_refs[0].get("display")

    return ProcedureOut(
        resource_id=doc["resource_id"],
        code=coding.get("code"),
        name=code_field.get("text") or coding.get("display"),
        system=coding.get("system"),
        status=data.get("status"),
        performed_date=performed_date,
        reason=reason,
    )


def normalize_encounter(doc: dict[str, Any]) -> EncounterOut:
    data = doc["data"]
    types = data.get("type")
    encounter_type = None
    if isinstance(types, list) and types and isinstance(types[0], dict):
        encounter_type = _coded_text(types[0], fallback_to_code=True)

    period = data.get("period") if isinstance(data.get("period"), dict) else {}

    reason = None
    reason_codes = data.get("reasonCode")
    if isinstance(reason_codes, list) and reason_codes:
        reason = _coded_text(reason_codes[0], fallback_to_code=True)

    location = None
    locations = data.get("location")
    if isinstance(locations, list) and locations and isinstance(locations[0], dict):
        location = (locations[0].get("location") or {}).get("display")

    return EncounterOut(
        resource_id=doc["resource_id"],
        encounter_type=encounter_type,
        status=data.get("status"),
        start_date=period.get("start"),
        end_date=period.get("end"),
        reason=reason,
        location=location,
    )


def normalize_diagnostic_report(doc: dict[str, Any]) -> DiagnosticReportOut:
    data = doc["data"]
    code_field = data.get("code") or {}

    category = None
    categories = data.get("category")
    if isinstance(categories, list) and categories and isinstance(categories[0], dict):
        category = _coded_text(categories[0], fallback_to_code=True)

    result_references = [
        strip_reference(result["reference"])
        for result in (data.get("result") or [])
        if isinstance(result, dict) and result.get("reference")
    ]

    return DiagnosticReportOut(
        resource_id=doc["resource_id"],
        report_name=code_field.get("text") or _first_coding(code_field).get("display"),
        status=data.get("status"),
        category=category,
        effective_date=data.get("effectiveDateTime"),
        result_references=result_references,
    )


def normalize_allergy(doc: dict[str, Any]) -> AllergyOut:
    data = doc["data"]
    code_field = data.get("code") or {}
    coding = _first_coding(code_field)

    reactions: list[AllergyReactionOut] = []
    all_manifestations: list[str] = []
    first_severity = None

    for reaction in data.get("reaction") or []:
        if not isinstance(reaction, dict):
            continue
        manifestations = [
            text
            for m in (reaction.get("manifestation") or [])
            if (text := _coded_text(m, fallback_to_code=True))
        ]
        severity = reaction.get("severity")
        reactions.append(AllergyReactionOut(manifestation=manifestations, severity=severity))
        all_manifestations.extend(manifestations)
        if first_severity is None:
            first_severity = severity

    return AllergyOut(
        resource_id=doc["resource_id"],
        substance=code_field.get("text") or coding.get("display"),
        code=coding.get("code"),
        system=coding.get("system"),
        clinical_status=_coded_text(data.get("clinicalStatus"), fallback_to_code=True),
        verification_status=_coded_text(data.get("verificationStatus"), fallback_to_code=True),
        reaction=reactions,
        severity=first_severity,
        manifestation=all_manifestations,
    )


# Maps a stored resource_type to (PatientProfile field name, normalizer).
# Adding a new normalized resource type later means adding one entry here.
_NORMALIZERS: dict[str, tuple[str, Any]] = {
    "Condition": ("conditions", normalize_condition),
    "Observation": ("observations", normalize_observation),
    "MedicationRequest": ("medications", normalize_medication),
    "Procedure": ("procedures", normalize_procedure),
    "Encounter": ("encounters", normalize_encounter),
    "DiagnosticReport": ("diagnostic_reports", normalize_diagnostic_report),
    "AllergyIntolerance": ("allergies", normalize_allergy),
}


def build_patient_profile(patient_doc: dict[str, Any], resource_docs: list[dict[str, Any]]) -> PatientProfileOut:
    """Assemble a PatientProfile from a patient document and that patient's
    FHIR resources. Resources tagged with a different patient_id are
    ignored, so a caller passing an unfiltered/incorrect resource list can
    never leak another patient's data into the profile."""
    patient_id = patient_doc["patient_id"]
    patient_data = patient_doc["data"]

    grouped: dict[str, list[Any]] = {field_name: [] for field_name, _ in _NORMALIZERS.values()}
    for doc in resource_docs:
        if doc.get("patient_id") != patient_id:
            continue
        mapping = _NORMALIZERS.get(doc.get("resource_type"))
        if mapping is None:
            continue
        field_name, normalizer = mapping
        grouped[field_name].append(normalizer(doc))

    return PatientProfileOut(
        patient_id=patient_id,
        demographics=normalize_demographics(patient_data),
        contact=normalize_contact(patient_data),
        normalized_at=datetime.now(timezone.utc),
        **grouped,
    )


# ---------------------------------------------------------------------------
# Service: batch/single-patient normalization against MongoDB.
# ---------------------------------------------------------------------------


@dataclass
class NormalizationStats:
    patients_processed: int = 0
    profiles_inserted: int = 0
    profiles_updated: int = 0
    profiles_failed: int = 0
    errors: list[str] = field(default_factory=list)


class PatientNormalizationService:
    """Builds and persists PatientProfiles from resources already stored by
    the Phase 1 ingestion pipeline."""

    def __init__(self, db: Database):
        self.db = db

    def normalize_patient(self, patient_id: str) -> Optional[PatientProfileOut]:
        """Normalize a single patient's resources and upsert the resulting
        profile. Returns None if the patient does not exist."""
        patient_doc = fhir_repository.get_patient(self.db, patient_id)
        if patient_doc is None:
            return None

        resource_docs = fhir_repository.get_all_patient_resources(self.db, patient_id)
        profile = build_patient_profile(patient_doc, resource_docs)
        patient_profile_repository.upsert_patient_profile(self.db, profile.model_dump())
        return profile

    def normalize_all(self) -> NormalizationStats:
        stats = NormalizationStats()
        for patient_id in fhir_repository.list_all_patient_ids(self.db):
            stats.patients_processed += 1
            try:
                patient_doc = fhir_repository.get_patient(self.db, patient_id)
                resource_docs = fhir_repository.get_all_patient_resources(self.db, patient_id)
                profile = build_patient_profile(patient_doc, resource_docs)
                result = patient_profile_repository.upsert_patient_profile(self.db, profile.model_dump())
                if result == "inserted":
                    stats.profiles_inserted += 1
                else:
                    stats.profiles_updated += 1
            except Exception as exc:  # e.g. pymongo errors, malformed stored resource
                stats.profiles_failed += 1
                stats.errors.append(f"{patient_id}: {exc}")

        return stats


def run_normalization() -> NormalizationStats:
    db = get_database()
    ensure_indexes(db)
    return PatientNormalizationService(db).normalize_all()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("Patient normalization started")
    stats = run_normalization()

    print()
    print(f"Patients processed: {stats.patients_processed}")
    print(f"Profiles inserted: {stats.profiles_inserted}")
    print(f"Profiles updated: {stats.profiles_updated}")
    print(f"Failed: {stats.profiles_failed}")

    if stats.errors:
        print()
        print(f"Errors ({len(stats.errors)}):")
        for error in stats.errors[:50]:
            print(f"  - {error}")
        if len(stats.errors) > 50:
            print(f"  ... and {len(stats.errors) - 50} more")

    print()
    print("Patient normalization completed")


if __name__ == "__main__":
    main()
