"""API response schemas. These wrap stored documents without altering the
preserved original FHIR JSON, which is always returned verbatim under `data`."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class FHIRResourceOut(BaseModel):
    resource_type: str
    resource_id: str
    patient_id: Optional[str] = None
    source_file: Optional[str] = None
    source_bundle_id: Optional[str] = None
    ingested_at: Optional[datetime] = None
    data: dict[str, Any]


class PatientOut(BaseModel):
    patient_id: str
    source_file: Optional[str] = None
    ingested_at: Optional[datetime] = None
    data: dict[str, Any]


class PaginationMeta(BaseModel):
    page: int
    page_size: int
    total: int
    total_pages: int


class PatientListOut(BaseModel):
    items: list[PatientOut]
    pagination: PaginationMeta


class ResourceListOut(BaseModel):
    items: list[FHIRResourceOut]
    pagination: PaginationMeta
