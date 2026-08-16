"""Hybrid Evidence retrieval: composes StructuredEvidenceRetriever and
SemanticEvidenceRetriever (both Phase 5A EvidenceRetriever implementations,
unmodified), deduplicates their combined Evidence, and re-ranks the result
with the existing DeterministicEvidenceRanker (Phase 5C, unmodified) — no
new ranking algorithm, no "hybrid confidence score", no LLM.

    RetrievalRequest
          |
    +-----+-----+
    |           |
Structured   Semantic
Retriever    Retriever
    |           |
    +-----+-----+
          |
      Evidence
          |
     Deduplicate
          |
  DeterministicEvidenceRanker
          |
        Top-K

Both underlying retrievers already scope every query to request.patient_id
at the source (never a cross-patient pool filtered afterward — see their
own module docstrings). This layer ADDITIONALLY, explicitly verifies that
every combined Evidence item's patient_id matches request.patient_id and
raises ValueError — never a silent filter — if it ever doesn't. This makes
a patient-scoping violation in either underlying retriever immediately
detectable rather than quietly hidden.

SCALABILITY: SemanticEvidenceRetriever is called with the same bounded
`limit` this method was asked for, and StructuredEvidenceRetriever already
scopes to one patient's resources of one known code (typically a handful,
verified up to a few dozen even in this project's real dataset) — neither
call is an unbounded fetch merely to satisfy a small top-K request.
"""

from __future__ import annotations

from typing import Optional

from pymongo.database import Database

from app.models.evidence import Evidence
from app.models.retrieval import RetrievalRequest, RetrievalResult
from app.services.evidence_ranker import DeterministicEvidenceRanker
from app.services.semantic_retriever import SemanticEvidenceRetriever
from app.services.structured_retriever import StructuredEvidenceRetriever

DEFAULT_LIMIT = 5


def _evidence_key(evidence: Evidence) -> tuple[str, str, str]:
    """Stable identity for deduplication: the same FHIR resource must
    appear only once in the final result, regardless of how many
    strategies retrieved it. Deliberately NOT based on display text, code
    alone, value, similarity, or object identity."""
    return (evidence.patient_id, evidence.resource_type, evidence.resource_id)


def _deduplicate(evidence: list[Evidence]) -> list[Evidence]:
    seen: set[tuple[str, str, str]] = set()
    deduplicated: list[Evidence] = []
    for item in evidence:
        key = _evidence_key(item)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)
    return deduplicated


class HybridEvidenceRetriever:
    """Composes the existing structured and semantic retrievers. Satisfies
    app.services.retrieval.EvidenceRetriever by structural typing
    (`retrieve(RetrievalRequest) -> RetrievalResult`; `limit` is an
    additional optional keyword argument, matching SemanticEvidenceRetriever)."""

    def __init__(self, db: Database, ranker: Optional[DeterministicEvidenceRanker] = None):
        self._structured = StructuredEvidenceRetriever(db)
        self._semantic = SemanticEvidenceRetriever(db)
        self._ranker = ranker or DeterministicEvidenceRanker()

    def retrieve(self, request: RetrievalRequest, *, limit: int = DEFAULT_LIMIT) -> RetrievalResult:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")

        structured_result = self._structured.retrieve(request)
        semantic_result = self._semantic.retrieve(request, limit=limit)

        if structured_result.status == "unsupported" and semantic_result.status == "unsupported":
            return RetrievalResult.unsupported(
                patient_id=request.patient_id,
                query=request.query,
                message="Neither structured nor semantic retrieval could service this query",
            )

        combined = _deduplicate(list(structured_result.evidence) + list(semantic_result.evidence))

        # Explicit, non-silent isolation check — a violation here means an
        # underlying retriever broke its own patient-scoping contract, and
        # that must be surfaced loudly, never quietly filtered away.
        mismatched = [item for item in combined if item.patient_id != request.patient_id]
        if mismatched:
            raise ValueError(
                f"HybridEvidenceRetriever received evidence for patient '{mismatched[0].patient_id}' while "
                f"servicing a request for patient '{request.patient_id}' — cross-patient evidence is not permitted"
            )

        if not combined:
            # Covers: both strategies found nothing, or one was unsupported
            # while the other simply found no evidence — either way, this
            # is an explicit "nothing found" state, never a negative fact.
            return RetrievalResult.from_evidence(patient_id=request.patient_id, query=request.query, evidence=[])

        # Phase 11B: when structured retrieval matched something (a
        # specific concept OR, since 11B-2, a category like "medications"),
        # every item it returns is already, by construction, of exactly one
        # resource_type — the one the query was actually asking about. That
        # resource_type is passed as a ranking hint so genuinely on-topic
        # structured matches aren't crowded out of the final top-K by
        # semantic hits that only coincidentally share vocabulary with a
        # DIFFERENT resource type (e.g. a "Medication Reconciliation"
        # Procedure outranking the patient's real MedicationRequest for a
        # "medications" query). This never changes WHAT is found — only
        # the deterministic tie-break already used elsewhere
        # (DeterministicEvidenceRanker) — and has no effect at all when
        # structured retrieval found nothing (the pre-existing behavior).
        resource_type_hint = structured_result.evidence[0].resource_type if structured_result.evidence else None

        ranked = self._ranker.rank(
            combined, patient_id=request.patient_id, resource_type_hint=resource_type_hint, limit=limit
        )
        return RetrievalResult.from_evidence(patient_id=request.patient_id, query=request.query, evidence=ranked)
