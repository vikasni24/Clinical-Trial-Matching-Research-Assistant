"""Data access layer for grounded-question audit records (Phase 7), stored
in the `audit_records` collection.

Every query here is patient-scoped by construction — there is no function
in this module that can produce an unscoped `find({})` scan across every
patient's audit history, and no function loads more than one page into
Python memory. Mirrors the existing pagination pattern already used by
app/repositories/evidence_repository.py and app/repositories/fhir_repository.py
(MongoDB-side skip/limit on an indexed, patient-scoped query).
"""

from __future__ import annotations

from typing import Any

from pymongo.database import Database

from app.db.mongodb import AUDIT_RECORDS_COLLECTION
from app.models.audit import AuditRecord

_EXCLUDE_MONGO_ID = {"_id": 0}


def insert_audit_record(db: Database, record: AuditRecord) -> None:
    """Persists one already-validated AuditRecord verbatim. This repository
    performs no transformation, inference, or enrichment of its own."""
    collection = db[AUDIT_RECORDS_COLLECTION]
    collection.insert_one(record.model_dump())


def get_patient_audit_history(
    db: Database, patient_id: str, page: int = 1, page_size: int = 20
) -> tuple[list[AuditRecord], int]:
    """A single page of one patient's audit history, newest first. Uses the
    idx_patient_created_at compound index for the scoped, sorted query and
    MongoDB-side skip/limit for pagination — never loads a patient's full
    audit history (let alone another patient's) into memory."""
    collection = db[AUDIT_RECORDS_COLLECTION]
    query: dict[str, Any] = {"patient_id": patient_id}

    skip = (page - 1) * page_size
    total = collection.count_documents(query)
    cursor = collection.find(query, _EXCLUDE_MONGO_ID).sort("created_at", -1).skip(skip).limit(page_size)
    return [AuditRecord(**document) for document in cursor], total
