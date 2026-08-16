"""Deterministic Evidence ranking/selection.

Orders and truncates an already-retrieved list of Evidence (e.g. from
StructuredEvidenceRetriever) for a future consumer such as Hybrid RAG
context construction — without ever claiming anything about clinical
truth. This module does NOT retrieve evidence and does NOT query
MongoDB: it operates purely on Evidence objects already in memory, so it
stays cheap and safe regardless of how large fhir_resources eventually
grows (ranking a list already bounded by the retrieval layer, never a
fresh database scan).

RANKING != TRUTH: the ordering signal computed below is informally called
"retrieval relevance" and is exclusively about which already-real Evidence
items to present first. A high rank never implies the underlying fact is
more likely true, that the patient currently has or lacks a condition, or
anything about eligibility — it is presentation ordering only, computed
from fields already present on Evidence (never inferred, generated, or
scored for medical confidence).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from app.models.evidence import Evidence

# rank(evidence) with no explicit limit returns at most this many items —
# enough for a typical context window without dumping a patient's entire
# history (e.g. all 12 historical HbA1c readings a real patient may have).
DEFAULT_LIMIT = 5

_FAR_FUTURE = datetime.max
_NO_DATE_SENTINEL = timedelta.max


def _parse_effective_date(effective_date: Optional[str]) -> Optional[datetime]:
    """Best-effort, crash-proof parse of Evidence.effective_date. Returns
    None for a missing or unparseable value — callers must treat None as
    "recency unknown", never as "now" and never invented."""
    if not effective_date:
        return None
    try:
        parsed = datetime.fromisoformat(effective_date)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        # Real stored dates mix naive ("2015-06-01") and timezone-aware
        # ("2001-12-21T03:14:47-05:00") strings; comparing them directly
        # would raise. Dropping the offset for ordering purposes only is a
        # deterministic normalization, not a claim about the clinical
        # timestamp itself.
        parsed = parsed.replace(tzinfo=None)
    return parsed


def _recency_key(effective_date: Optional[str]) -> timedelta:
    """Smaller = more recent, for use in an ascending sort. Missing or
    malformed dates always sort last — never assumed to be current."""
    parsed = _parse_effective_date(effective_date)
    if parsed is None:
        return _NO_DATE_SENTINEL
    return _FAR_FUTURE - parsed


def _retrieval_relevance_key(evidence: Evidence, resource_type_hint: Optional[str]) -> tuple:
    """The deterministic ordering-only signal used to sort Evidence. This
    is NOT a measure of clinical truth, certainty, or eligibility — purely
    which already-real evidence item should be presented first. Smaller
    tuples sort first (ascending sort = highest relevance first):

      1. has a code (fully identified evidence before partially-identified)
      2. matches the caller's resource_type hint, if given
      3. carries a recorded value
      4. more recent effective_date (missing/malformed dates sort last)
      5. (resource_type, resource_id) — stable, deterministic tie-break
    """
    has_code = evidence.code is not None
    matches_resource_type = resource_type_hint is not None and evidence.resource_type == resource_type_hint
    has_value = evidence.value is not None
    return (
        0 if has_code else 1,
        0 if matches_resource_type else 1,
        0 if has_value else 1,
        _recency_key(evidence.effective_date),
        evidence.resource_type,
        evidence.resource_id,
    )


class DeterministicEvidenceRanker:
    """Stateless — no MongoDB/service dependency, no constructor arguments.
    Operates only on Evidence objects a caller already retrieved (e.g. via
    StructuredEvidenceRetriever); never fetches additional data itself."""

    def rank(
        self,
        evidence: list[Evidence],
        *,
        patient_id: Optional[str] = None,
        resource_type_hint: Optional[str] = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[Evidence]:
        """Returns a deterministically ordered, truncated copy of
        `evidence`. Never mutates, fabricates, or drops fields from any
        Evidence object — only reorders and (optionally) truncates the
        list itself.

        patient_id is optional: this layer trusts that its caller (e.g.
        StructuredEvidenceRetriever, itself patient-scoped via
        EvidenceService) already constructed a correctly-scoped evidence
        list. When patient_id IS supplied, it is used as a defensive
        check — every item must already belong to that patient, or a
        ValueError is raised rather than silently mixing evidence across
        patients.
        """
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")

        if patient_id is not None:
            mismatched = [item for item in evidence if item.patient_id != patient_id]
            if mismatched:
                raise ValueError(
                    f"Evidence for patient '{mismatched[0].patient_id}' cannot be ranked under "
                    f"patient_id='{patient_id}' — cross-patient evidence is not permitted"
                )

        if not evidence:
            return []

        ranked = sorted(evidence, key=lambda item: _retrieval_relevance_key(item, resource_type_hint))
        return ranked[:limit]
