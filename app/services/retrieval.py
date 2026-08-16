"""The provider-independent retrieval INTERFACE a future Hybrid RAG system's
retrieval strategies must implement.

NO strategy is implemented here — this module defines only the contract.
Phase 5B+ may add deterministic-structured, keyword, vector, graph, or
hybrid implementations of EvidenceRetriever; each must accept a
RetrievalRequest and return a RetrievalResult (app/models/retrieval.py),
so downstream RAG/LLM code can depend on this Protocol alone and remain
unaware of which strategy (or combination) is actually answering a given
query.

Whatever a future implementation does internally, the same rule applies:
the source of truth stays MongoDB -> fhir_resources -> the Evidence
Foundation (app/repositories/evidence_repository.py,
app/services/evidence_service.py). An EvidenceRetriever may retrieve
evidence differently, but it must never invent it.
"""

from __future__ import annotations

from typing import Protocol

from app.models.retrieval import RetrievalRequest, RetrievalResult


class EvidenceRetriever(Protocol):
    """A future retrieval strategy. No implementation exists yet."""

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult: ...
