"""Phase 7: AuditService — records and retrieves the deterministic audit
trail for POST /api/patients/{patient_id}/ask.

This service performs no eligibility reasoning, retrieves no clinical
evidence of its own, never calls an LLM, and never modifies a
GroundedAnswer. It only records facts the pipeline has ALREADY established
(GroundedContext.status, GroundedAnswer.status, and the Evidence actually
attached to the final validated answer) and retrieves them back,
patient-scoped and paginated. `evidence_references` are built exclusively
from `answer.evidence` — the Evidence objects the final, already-validated
GroundedAnswer actually carries — never from context.evidence directly, so
an audit record can never describe evidence broader than what the client
actually received. When `answer.status != "answered"`, `answer.evidence` is
structurally guaranteed empty by GroundedAnswer's own validator
(app/models/answer.py), so the resulting audit record's evidence_references
is correctly empty too — no re-derivation needed here.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pymongo.database import Database

from app.models.answer import GroundedAnswer
from app.models.audit import AuditEvidenceReference, AuditRecord
from app.models.rag_context import GroundedContext
from app.repositories import audit_repository


class AuditService:
    def __init__(self, db: Database):
        self.db = db

    def record_question_audit(self, context: GroundedContext, answer: GroundedAnswer) -> AuditRecord:
        """Builds and persists one AuditRecord from an already-produced,
        already-validated GroundedContext + GroundedAnswer pair. Generates a
        fresh audit_id and created_at timestamp; every other field is a
        direct copy from `context`/`answer`."""
        evidence_references = [
            AuditEvidenceReference(
                patient_id=item.patient_id, resource_type=item.resource_type, resource_id=item.resource_id
            )
            for item in answer.evidence
        ]
        record = AuditRecord(
            audit_id=str(uuid.uuid4()),
            patient_id=answer.patient_id,
            query=answer.query,
            retrieval_status=context.status,
            answer_status=answer.status,
            evidence_references=evidence_references,
            created_at=datetime.now(timezone.utc),
        )
        audit_repository.insert_audit_record(self.db, record)
        return record

    def get_patient_audit_history(
        self, patient_id: str, page: int = 1, page_size: int = 20
    ) -> tuple[list[AuditRecord], int]:
        return audit_repository.get_patient_audit_history(self.db, patient_id, page=page, page_size=page_size)
