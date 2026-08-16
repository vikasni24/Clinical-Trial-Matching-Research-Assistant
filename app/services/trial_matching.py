"""Orchestrates deterministic patient-vs-trial eligibility matching.

Retrieves a single normalized PatientProfile and either one trial or a
filtered candidate set of trials from MongoDB, then delegates the actual
PASS/FAIL/UNKNOWN evaluation to EligibilityMatcher. No LLM, no embeddings,
no RAG, and no inference beyond what EligibilityMatcher already performs
deterministically from explicit fields already present in the profile and
trial documents.

SCALABILITY NOTE: at most one PatientProfile is ever loaded into memory per
call — never the full patients collection. The trial candidate set is
retrieved via trial_repository.iter_candidate_trials, an indexed, lazily
-evaluated MongoDB cursor (filtered on `status`, covered by idx_trial_status)
— trial documents are evaluated and converted to MatchResultOut one at a
time rather than the full candidate set being materialized into memory up
front. This lets the trial catalog grow independently of matcher memory use.

EVIDENCE (Phase 4D): the matcher is constructed with an EvidenceService
bound to this same `db`, so criterion evaluations can attach traceable
Evidence (see app/services/eligibility_matcher.py). This adds a small,
bounded number of additional targeted MongoDB lookups per match — one per
criterion that a specific resource backed — never a collection scan.
"""

from __future__ import annotations

from typing import Optional

from pymongo.database import Database

from app.models.clinical_trial import ClinicalTrialOut
from app.models.match_result import MatchResultOut
from app.models.patient_profile import PatientProfileOut
from app.repositories import patient_profile_repository, trial_repository
from app.services.eligibility_matcher import EligibilityMatcher
from app.services.evidence_service import EvidenceService

DEFAULT_CANDIDATE_STATUS = "recruiting"

# ELIGIBLE ranks first, then UNKNOWN, then INELIGIBLE; ties broken by match_score.
_STATUS_RANK = {"ELIGIBLE": 0, "UNKNOWN": 1, "INELIGIBLE": 2}


class TrialMatchingService:
    def __init__(self, db: Database):
        self.db = db
        self._matcher = EligibilityMatcher(evidence_service=EvidenceService(db))

    def match_patient_to_trial(self, patient_id: str, trial_id: str) -> Optional[MatchResultOut]:
        """Evaluate one patient against one trial. Returns None if either
        the patient's profile or the trial cannot be found. Callers that
        need to distinguish "patient not found" from "trial not found" for
        an accurate 404 message should check existence themselves first
        (see the API routes, which follow the same pattern already used by
        GET /api/patients/{id}/resources in Phase 1)."""
        profile_doc = patient_profile_repository.get_patient_profile(self.db, patient_id)
        if profile_doc is None:
            return None

        trial_doc = trial_repository.get_trial(self.db, trial_id)
        if trial_doc is None:
            return None

        return self._matcher.evaluate(PatientProfileOut(**profile_doc), ClinicalTrialOut(**trial_doc))

    def match_patient_to_trials(
        self, patient_id: str, status: Optional[str] = DEFAULT_CANDIDATE_STATUS
    ) -> Optional[list[MatchResultOut]]:
        """Evaluate one patient against every candidate trial. By default
        only trials with status == "recruiting" are considered candidates
        (basic deterministic filtering per Phase 3 Step 13); pass
        status=None to evaluate every trial regardless of recruitment
        status. Returns None if the patient has no PatientProfile."""
        profile_doc = patient_profile_repository.get_patient_profile(self.db, patient_id)
        if profile_doc is None:
            return None
        profile = PatientProfileOut(**profile_doc)

        # Each trial document is converted to a MatchResultOut and discarded
        # as the cursor is consumed — the raw candidate set is never held in
        # memory in full, only the (much smaller) results being ranked below.
        results = [
            self._matcher.evaluate(profile, ClinicalTrialOut(**doc))
            for doc in trial_repository.iter_candidate_trials(self.db, status=status)
        ]

        results.sort(key=lambda r: (_STATUS_RANK[r.eligibility_status], -r.match_score))
        return results
