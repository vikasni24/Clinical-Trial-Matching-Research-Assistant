"""Deterministic evidence retrieval service — the clean interface future
consumers (Phase 3's eligibility matcher, and later RAG/Research Assistant
components) will use to fetch factual clinical evidence.

Architecture: API -> Service -> Repository -> MongoDB. This service never
queries MongoDB directly; every method here delegates straight to
app/repositories/evidence_repository.py, which owns all query/index
knowledge for the fhir_resources collection. The service adds a stable,
orchestration-only interface — no inference, no scoring, no generation,
and no MongoDB query logic of its own. If evidence isn't in MongoDB, it
does not exist as far as this service is concerned; "no evidence found" is
represented as an empty iterable or None, never as a guess.
"""

from __future__ import annotations

from typing import Iterator, Optional

from pymongo.database import Database

from app.models.evidence import Evidence
from app.repositories import evidence_repository


class EvidenceService:
    """Thin orchestration layer over evidence_repository. Mirrors the
    existing TrialMatchingService pattern: one instance bound to a single
    `db` handle, methods delegate to repository functions rather than
    building queries themselves."""

    def __init__(self, db: Database):
        self.db = db

    def get_patient_evidence(self, patient_id: str, resource_type: Optional[str] = None) -> Iterator[Evidence]:
        """All of a patient's evidence, optionally filtered to one resource
        type. Lazy — see evidence_repository.get_patient_evidence."""
        return evidence_repository.get_patient_evidence(self.db, patient_id, resource_type=resource_type)

    def get_patient_observation_evidence(self, patient_id: str) -> Iterator[Evidence]:
        """A patient's Observation evidence only."""
        return evidence_repository.get_patient_observation_evidence(self.db, patient_id)

    def get_patient_evidence_by_code(
        self, patient_id: str, code: str, resource_type: Optional[str] = None
    ) -> Iterator[Evidence]:
        """Evidence for a patient's resources whose primary coding matches
        `code`, optionally narrowed to one resource type."""
        return evidence_repository.get_patient_evidence_by_code(
            self.db, patient_id, code, resource_type=resource_type
        )

    def get_patient_resource_evidence(
        self, patient_id: str, resource_type: str, resource_id: str
    ) -> Optional[Evidence]:
        """A single resource's evidence, scoped to exactly this patient.
        None if no such resource exists for this patient."""
        return evidence_repository.get_patient_resource_evidence(self.db, patient_id, resource_type, resource_id)
