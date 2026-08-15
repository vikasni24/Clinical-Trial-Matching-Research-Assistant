"""Parsing of standalone FHIR resources and FHIR Bundles.

Extensible by design: to support an additional resource type later, add it to
SUPPORTED_RESOURCE_TYPES. Everything else (Bundle extraction, id/patient
association, error reporting) is generic and does not need to change.
"""

from dataclasses import dataclass
from typing import Any, Optional

SUPPORTED_RESOURCE_TYPES = frozenset(
    {
        "Patient",
        "Observation",
        "Condition",
        "MedicationRequest",
        "Procedure",
        "Encounter",
        "DiagnosticReport",
        "AllergyIntolerance",
    }
)


@dataclass
class ParsedResource:
    resource_type: str
    resource_id: str
    patient_id: Optional[str]
    data: dict[str, Any]
    source_bundle_id: Optional[str] = None


@dataclass
class ResourceOutcome:
    status: str  # "parsed" | "skipped_unsupported" | "failed"
    resource: Optional[ParsedResource] = None
    message: Optional[str] = None


def strip_reference(reference: str) -> str:
    """Extract the bare resource id from a FHIR reference string, which may be
    expressed as "ResourceType/id" or as "urn:uuid:id"."""
    if reference.startswith("urn:uuid:"):
        return reference[len("urn:uuid:") :]
    return reference.split("/")[-1] if "/" in reference else reference


def extract_patient_id(resource: dict[str, Any]) -> Optional[str]:
    """Best-effort patient association via the resource itself (Patient) or
    its subject/patient reference, which Synthea/FHIR may express either as
    "Patient/patient-001" or as "urn:uuid:patient-001"."""
    if resource.get("resourceType") == "Patient":
        return resource.get("id")

    reference_container = resource.get("subject") or resource.get("patient")
    if not isinstance(reference_container, dict):
        return None

    reference = reference_container.get("reference")
    if not reference:
        return None

    return strip_reference(reference)


def _parse_single_resource(
    resource: Any, source_file: str, bundle_id: Optional[str]
) -> ResourceOutcome:
    if not isinstance(resource, dict):
        return ResourceOutcome(
            status="failed",
            message=f"{source_file}: entry is not a valid FHIR resource object",
        )

    resource_type = resource.get("resourceType")
    if resource_type not in SUPPORTED_RESOURCE_TYPES:
        return ResourceOutcome(
            status="skipped_unsupported",
            message=f"{source_file}: unsupported resourceType '{resource_type}', skipped",
        )

    resource_id = resource.get("id")
    if not resource_id:
        return ResourceOutcome(
            status="failed",
            message=f"{source_file}: {resource_type} resource missing 'id', skipped",
        )

    return ResourceOutcome(
        status="parsed",
        resource=ParsedResource(
            resource_type=resource_type,
            resource_id=resource_id,
            patient_id=extract_patient_id(resource),
            data=resource,
            source_bundle_id=bundle_id,
        ),
    )


def extract_resources(raw: Any, source_file: str) -> list[ResourceOutcome]:
    """Extract one or more resource outcomes from a parsed JSON document,
    which may be a standalone FHIR resource or a Bundle."""
    if not isinstance(raw, dict):
        return [
            ResourceOutcome(
                status="failed",
                message=f"{source_file}: root JSON is not a FHIR resource object",
            )
        ]

    if raw.get("resourceType") == "Bundle":
        bundle_id = raw.get("id")
        entries = raw.get("entry")
        if not isinstance(entries, list):
            return [
                ResourceOutcome(
                    status="failed",
                    message=f"{source_file}: Bundle has no valid 'entry' list",
                )
            ]

        outcomes: list[ResourceOutcome] = []
        for index, entry in enumerate(entries):
            inner = entry.get("resource") if isinstance(entry, dict) else None
            if inner is None:
                outcomes.append(
                    ResourceOutcome(
                        status="failed",
                        message=f"{source_file}: bundle entry {index} missing 'resource'",
                    )
                )
                continue
            outcomes.append(_parse_single_resource(inner, source_file, bundle_id))
        return outcomes

    return [_parse_single_resource(raw, source_file, None)]
