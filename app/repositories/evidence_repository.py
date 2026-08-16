"""Targeted, deterministic retrieval of Evidence from the existing
`fhir_resources` collection.

This repository does NOT create or persist a new collection — Evidence
objects are derived on demand from the FHIR resources Phase 1 already
stored, which remain the untouched source of truth. Every query here is
scoped by patient_id (and usually resource_type), using the same
`idx_patient_resource_type` compound index Phase 1 already created — no
function in this module loads the full collection, another patient's
resources, or an unbounded result set into Python.

Extraction is purely mechanical FHIR field lookup (mirrors the same
deterministic approach used by app/services/patient_normalization.py and
app/services/eligibility_matcher.py): no LLM, no semantic/fuzzy matching,
no inferred values. A field that isn't present in the stored FHIR JSON is
left None on the resulting Evidence object.
"""

from typing import Any, Iterator, Optional

from pymongo.database import Database

from app.db.mongodb import FHIR_RESOURCES_COLLECTION
from app.models.evidence import Evidence

# Only the fields actually needed to build Evidence are returned — avoids
# pulling source_file/source_bundle_id/ingested_at across the wire for
# every matching document.
_EVIDENCE_PROJECTION = {"_id": 0, "patient_id": 1, "resource_type": 1, "resource_id": 1, "data": 1}

# Where a resource type's primary CodeableConcept lives. Most FHIR R4
# resources use `code`; MedicationRequest and Encounter are the two
# supported exceptions.
_CODE_FIELD_BY_RESOURCE_TYPE = {
    "MedicationRequest": "medicationCodeableConcept",
    "Encounter": "type",  # a list of CodeableConcept, not a single one — handled separately
}

_CODE_PATH_BY_RESOURCE_TYPE = {
    "Condition": "data.code.coding.code",
    "Observation": "data.code.coding.code",
    "AllergyIntolerance": "data.code.coding.code",
    "Procedure": "data.code.coding.code",
    "DiagnosticReport": "data.code.coding.code",
    "MedicationRequest": "data.medicationCodeableConcept.coding.code",
    "Encounter": "data.type.coding.code",
}

_VALUE_FIELDS = ("valueQuantity", "valueInteger", "valueDecimal", "valueString", "valueCodeableConcept")


def _first_coding(codeable_concept: Optional[dict]) -> dict:
    if not isinstance(codeable_concept, dict):
        return {}
    codings = codeable_concept.get("coding")
    if isinstance(codings, list) and codings and isinstance(codings[0], dict):
        return codings[0]
    return {}


def _coded_text(codeable_concept: Optional[dict], fallback_to_code: bool = False) -> Optional[str]:
    """Best human-readable label for a CodeableConcept: `.text`, else the
    primary coding's `.display`, else (optionally) its bare `.code`. Status
    codes like "active"/"confirmed" often carry no separate display/text in
    stored FHIR data but are already human-readable, so status extraction
    below opts into the code fallback — this surfaces real recorded data,
    it does not invent anything."""
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


def _extract_value(data: dict[str, Any]) -> tuple[Optional[Any], Optional[str]]:
    for field_name in _VALUE_FIELDS:
        if field_name not in data:
            continue
        raw = data[field_name]
        if field_name == "valueQuantity" and isinstance(raw, dict):
            return raw.get("value"), raw.get("unit")
        if field_name == "valueCodeableConcept":
            return _coded_text(raw), None
        return raw, None
    return None, None


def _extract_component_summary(data: dict[str, Any]) -> Optional[str]:
    """Phase 11B-1: FHIR panel-style Observations (e.g. Blood Pressure,
    LOINC 85354-9) carry no top-level value[x] at all — the actual
    measurements live in `component[].valueQuantity`, one per sub-reading
    (e.g. Systolic 8480-6 / Diastolic 8462-4). `_extract_value` above never
    looks there, so these were previously silently dropped (Evidence.value
    stayed None even though real, stored values existed).

    This deterministically builds a single human-readable, LOINC-labeled
    summary string from each component's own code/display + valueQuantity
    — e.g. "Systolic Blood Pressure: 125 mm[Hg]; Diastolic Blood Pressure:
    85 mm[Hg]" — so systolic and diastolic (or any other panel's
    components) stay distinguishable from each other within the existing
    Evidence.value field. Only ever reads component code/display/
    valueQuantity/valueCodeableConcept/valueString — the same class of
    fields already extracted elsewhere in this module; never the raw FHIR
    document itself, and no new Evidence field is introduced."""
    components = data.get("component")
    if not isinstance(components, list):
        return None

    parts: list[str] = []
    for component in components:
        if not isinstance(component, dict):
            continue
        code_field = component.get("code") or {}
        coding = _first_coding(code_field)
        label = code_field.get("text") or coding.get("display") or coding.get("code")

        quantity = component.get("valueQuantity")
        if isinstance(quantity, dict) and quantity.get("value") is not None:
            value_text = str(quantity["value"])
            if quantity.get("unit"):
                value_text += f" {quantity['unit']}"
        else:
            value_text = _coded_text(component.get("valueCodeableConcept")) or component.get("valueString")

        if label and value_text:
            parts.append(f"{label}: {value_text}")

    return "; ".join(parts) if parts else None


def _extract_code_fields(data: dict[str, Any], resource_type: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Returns (code, coding_system, display) from the resource's primary
    CodeableConcept, whichever field that lives under for this resource type."""
    if resource_type == "Encounter":
        types = data.get("type")
        concept = types[0] if isinstance(types, list) and types and isinstance(types[0], dict) else None
    else:
        field_name = _CODE_FIELD_BY_RESOURCE_TYPE.get(resource_type, "code")
        concept = data.get(field_name)

    coding = _first_coding(concept)
    display = _coded_text(concept)
    return coding.get("code"), coding.get("system"), display


def _document_to_evidence(document: dict[str, Any]) -> Evidence:
    """Deterministically converts one stored fhir_resources document into an
    Evidence object. Never invents a field the source FHIR JSON lacks."""
    resource_type = document["resource_type"]
    data = document.get("data") or {}

    code: Optional[str] = None
    coding_system: Optional[str] = None
    display: Optional[str] = None
    value: Optional[Any] = None
    unit: Optional[str] = None
    effective_date: Optional[str] = None
    status: Optional[str] = None

    if resource_type == "Patient":
        given = data.get("name", [{}])[0].get("given") if data.get("name") else None
        family = data.get("name", [{}])[0].get("family") if data.get("name") else None
        name_parts = [p for p in [*(given or []), family] if p]
        display = " ".join(name_parts) if name_parts else None
        value = data.get("gender")
        effective_date = data.get("birthDate")

    elif resource_type == "Observation":
        code, coding_system, display = _extract_code_fields(data, resource_type)
        value, unit = _extract_value(data)
        if value is None:
            # No top-level value[x] — check for a panel/component structure
            # (e.g. Blood Pressure) before leaving this genuinely unset.
            value = _extract_component_summary(data)
        effective_date = data.get("effectiveDateTime")
        status = data.get("status")

    elif resource_type == "Condition":
        code, coding_system, display = _extract_code_fields(data, resource_type)
        status = _coded_text(data.get("clinicalStatus"), fallback_to_code=True)
        effective_date = data.get("onsetDateTime") or data.get("recordedDate")

    elif resource_type == "MedicationRequest":
        code, coding_system, display = _extract_code_fields(data, resource_type)
        status = data.get("status")
        effective_date = data.get("authoredOn")

    elif resource_type == "AllergyIntolerance":
        code, coding_system, display = _extract_code_fields(data, resource_type)
        status = _coded_text(data.get("clinicalStatus"), fallback_to_code=True)
        effective_date = data.get("recordedDate")

    elif resource_type == "Procedure":
        code, coding_system, display = _extract_code_fields(data, resource_type)
        status = data.get("status")
        effective_date = data.get("performedDateTime")
        if not effective_date and isinstance(data.get("performedPeriod"), dict):
            effective_date = data["performedPeriod"].get("start")

    elif resource_type == "Encounter":
        code, coding_system, display = _extract_code_fields(data, resource_type)
        status = data.get("status")
        if isinstance(data.get("period"), dict):
            effective_date = data["period"].get("start")

    elif resource_type == "DiagnosticReport":
        code, coding_system, display = _extract_code_fields(data, resource_type)
        status = data.get("status")
        effective_date = data.get("effectiveDateTime")

    # Any other resource type: traceability fields only, every clinical
    # field stays None rather than guessing at an unfamiliar structure.

    return Evidence(
        patient_id=document["patient_id"],
        resource_type=resource_type,
        resource_id=document["resource_id"],
        source_collection=FHIR_RESOURCES_COLLECTION,
        code=code,
        coding_system=coding_system,
        display=display,
        value=value,
        unit=unit,
        effective_date=effective_date,
        status=status,
    )


def _build_query(patient_id: str, resource_type: Optional[str] = None, code: Optional[str] = None) -> dict[str, Any]:
    """The shared patient-scoped Mongo filter used by every evidence query
    below. When `code` is given: if resource_type is also known, the single
    corresponding coding path (e.g. `data.medicationCodeableConcept.coding.code`
    for MedicationRequest) is queried directly; otherwise every known coding
    path is checked via $or — still always scoped to this patient only."""
    query: dict[str, Any] = {"patient_id": patient_id}
    if resource_type:
        query["resource_type"] = resource_type

    if code:
        if resource_type:
            code_path = _CODE_PATH_BY_RESOURCE_TYPE.get(resource_type, "data.code.coding.code")
            query[code_path] = code
        else:
            distinct_paths = sorted(set(_CODE_PATH_BY_RESOURCE_TYPE.values()))
            query["$or"] = [{path: code} for path in distinct_paths]

    return query


def get_patient_evidence(db: Database, patient_id: str, resource_type: Optional[str] = None) -> Iterator[Evidence]:
    """Lazily yields Evidence for every one of a patient's FHIR resources,
    optionally filtered to a single resource_type. Scoped by patient_id (and
    resource_type when given) via the existing idx_patient_resource_type
    index — never touches another patient's resources or the full
    collection."""
    collection = db[FHIR_RESOURCES_COLLECTION]
    query = _build_query(patient_id, resource_type=resource_type)
    for document in collection.find(query, _EVIDENCE_PROJECTION):
        yield _document_to_evidence(document)


def get_patient_observation_evidence(db: Database, patient_id: str) -> Iterator[Evidence]:
    """Evidence for a patient's Observation resources only."""
    return get_patient_evidence(db, patient_id, resource_type="Observation")


def get_patient_evidence_by_code(
    db: Database, patient_id: str, code: str, resource_type: Optional[str] = None
) -> Iterator[Evidence]:
    """Lazily yields Evidence for a patient's resources whose primary coding
    contains the given code. Filtering happens entirely in MongoDB."""
    collection = db[FHIR_RESOURCES_COLLECTION]
    query = _build_query(patient_id, resource_type=resource_type, code=code)
    for document in collection.find(query, _EVIDENCE_PROJECTION):
        yield _document_to_evidence(document)


def list_patient_evidence(
    db: Database,
    patient_id: str,
    resource_type: Optional[str] = None,
    code: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[Evidence], int]:
    """Paginated Evidence retrieval for API browsing. Unlike the
    generator-based functions above (meant for internal bulk/streaming
    use, e.g. by the eligibility matcher), this is for a single page of
    results and uses MongoDB-side skip/limit — mirrors the existing
    fhir_repository.list_patient_resources pattern exactly, so only
    page_size documents are ever transferred, never the full result set."""
    collection = db[FHIR_RESOURCES_COLLECTION]
    query = _build_query(patient_id, resource_type=resource_type, code=code)

    skip = (page - 1) * page_size
    total = collection.count_documents(query)
    cursor = (
        collection.find(query, _EVIDENCE_PROJECTION)
        .sort([("resource_type", 1), ("resource_id", 1)])
        .skip(skip)
        .limit(page_size)
    )
    items = [_document_to_evidence(document) for document in cursor]
    return items, total


def get_patient_resource_evidence(
    db: Database, patient_id: str, resource_type: str, resource_id: str
) -> Optional[Evidence]:
    """A single resource's Evidence, scoped to exactly this patient — used
    when a specific resource_id is already known (e.g. tracing a criterion
    evaluation back to the resource that produced it). Returns None if no
    such resource exists for this patient (including if the resource_id
    belongs to a different patient)."""
    collection = db[FHIR_RESOURCES_COLLECTION]
    document = collection.find_one(
        {"patient_id": patient_id, "resource_type": resource_type, "resource_id": resource_id},
        _EVIDENCE_PROJECTION,
    )
    if document is None:
        return None
    return _document_to_evidence(document)
