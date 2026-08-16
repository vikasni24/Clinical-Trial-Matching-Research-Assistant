"""Packages a retrieval result into GroundedContext for a future Phase 6
(LLM) consumer.

    query
      |
HybridEvidenceRetriever
      |
  RetrievalResult
      |
  GroundedContext

This module does NOT call an LLM and performs no additional retrieval,
inference, or generation of its own — `build_grounded_context` is a pure,
deterministic field-for-field repackaging of an already-computed
RetrievalResult; `GroundedContextService` is a thin convenience that chains
HybridEvidenceRetriever with it.
"""

from __future__ import annotations

from pymongo.database import Database

from app.models.rag_context import GroundedContext
from app.models.retrieval import RetrievalRequest, RetrievalResult
from app.services.hybrid_retriever import DEFAULT_LIMIT, HybridEvidenceRetriever


def build_grounded_context(result: RetrievalResult) -> GroundedContext:
    """Deterministically repackages a RetrievalResult as GroundedContext.
    Every field is copied directly from `result`; nothing is added,
    generated, inferred, or guessed."""
    return GroundedContext(
        patient_id=result.patient_id,
        query=result.query,
        status=result.status,
        evidence=result.evidence,
        message=result.message,
    )


class GroundedContextService:
    """Convenience chain: HybridEvidenceRetriever -> GroundedContext."""

    def __init__(self, db: Database):
        self._retriever = HybridEvidenceRetriever(db)

    def build_context(self, request: RetrievalRequest, *, limit: int = DEFAULT_LIMIT) -> GroundedContext:
        result = self._retriever.retrieve(request, limit=limit)
        return build_grounded_context(result)
