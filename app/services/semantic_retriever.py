"""Deterministic, dependency-free semantic-ish retrieval — the second
EvidenceRetriever (Phase 5A protocol) implementation, alongside Phase 5B's
StructuredEvidenceRetriever.

EMBEDDING APPROACH: this is deliberately NOT a neural/API embedding. It is
a classic, fully local, fully deterministic bag-of-words term-frequency
vector with cosine similarity — chosen per this phase's own guidance to
prefer a "deterministic local semantic representation" over either (a) a
heavy ML dependency (e.g. sentence-transformers/torch) that would slow
down and destabilize this project's lightweight test environment, or (b)
an external hosted-LLM embedding API, which would require credentials and
a network dependency this project has consistently avoided. It matches a
query to evidence purely by shared vocabulary (after removing a small
fixed stopword list) — it has no understanding of medical synonyms (e.g.
"HbA1c" and "Hemoglobin A1c" only match if they literally share a word)
and makes no claim otherwise.

Retrieval stays patient-scoped at the SOURCE query, not via post-filtering:
EvidenceService.get_patient_evidence(patient_id) is called once per
request and only ever returns that one patient's resources — there is no
global/cross-patient index anywhere in this module.

Ranking is a separate, reused concern: DeterministicEvidenceRanker's own
public `rank()` (Phase 5C, unmodified) breaks ties among evidence with
identical similarity, so its (resource_type, resource_id) tie-break isn't
reimplemented here.

NO EVIDENCE != NEGATIVE FACT: a query below the relevance threshold, a
patient with no evidence at all, or a query with no meaningful terms all
return an explicit non-assertive state (`no_evidence_found` or
`unsupported`) with evidence=[] — never a fabricated item, never a
clinical claim, never a confidence/probability score.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from pymongo.database import Database

from app.models.evidence import Evidence
from app.models.retrieval import RetrievalRequest, RetrievalResult
from app.services.evidence_ranker import DeterministicEvidenceRanker
from app.services.evidence_service import EvidenceService
from app.services.structured_retriever import CODE_TO_CONCEPT_KEY, StructuredConcept, _match_concept

DEFAULT_LIMIT = 5

# Below this cosine similarity, evidence is considered not relevant enough
# to return. This is purely a retrieval-relevance cutoff — never a medical
# confidence threshold — deterministic and easy to adjust later. Phase 11B
# diagnosis confirmed this value is not the cause of missed category-style
# matches (every failing case scored exactly 0.0 — zero token overlap, not
# a near-miss) — see _PLURAL_TO_SINGULAR and _is_registry_conflict below
# for the actual, deterministic fixes; this threshold is unchanged.
MIN_RELEVANCE_THRESHOLD = 0.1

# A small, fixed, deterministic stopword list — not a language model, just
# common query filler that would otherwise dilute lexical overlap.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
        "for", "from", "had", "has", "have", "he", "her", "his", "how",
        "in", "is", "it", "of", "on", "or", "patient", "patients", "s",
        "she", "that", "the", "their", "this", "to", "was", "were",
        "what", "when", "where", "which", "who", "why", "with",
    }
)

# Phase 11B-3: a small, explicit, hand-curated plural -> singular mapping —
# deliberately NOT a general stemming rule (e.g. "strip trailing s"), which
# would mangle words like "status" -> "statu" or "diabetes" -> "diabete".
# Only the specific irregular/regular plurals this system's own vocabulary
# actually uses are listed; anything not listed here passes through
# unchanged, so unrelated words can never be damaged by this step.
_PLURAL_TO_SINGULAR = {
    "medications": "medication",
    "medicines": "medicine",
    "meds": "medication",
    "drugs": "drug",
    "conditions": "condition",
    "diagnoses": "diagnosis",
    "allergies": "allergy",
    "symptoms": "symptom",
    "diseases": "disease",
    "disorders": "disorder",
    "procedures": "procedure",
    "observations": "observation",
    "readings": "reading",
    "results": "result",
    "levels": "level",
    "values": "value",
}

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _normalize_token(token: str) -> str:
    """Deterministic, explicit-allowlist-only normalization — see
    _PLURAL_TO_SINGULAR. A token not in the map is returned unchanged."""
    return _PLURAL_TO_SINGULAR.get(token, token)


def _tokenize(text: str) -> Counter:
    """Deterministic lowercase word tokenization with the fixed stopword
    list removed and known plural forms normalized. No stemming library, no
    synonym expansion, no language model — see _PLURAL_TO_SINGULAR."""
    tokens = _TOKEN_PATTERN.findall(text.lower())
    return Counter(_normalize_token(token) for token in tokens if token not in _STOPWORDS)


def _evidence_text(evidence: Evidence) -> str:
    """Searchable text for one Evidence item, built ONLY from its existing
    structured fields — never the raw FHIR document, never _id, never
    patient_id/resource_id/source_reference (identifiers aren't clinical
    vocabulary)."""
    parts = [
        evidence.resource_type,
        evidence.code,
        evidence.coding_system,
        evidence.display,
        str(evidence.value) if evidence.value is not None else None,
        evidence.unit,
        evidence.status,
        evidence.effective_date,
    ]
    return " ".join(part for part in parts if part)


def _cosine_similarity(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[token] * b[token] for token in common)
    if dot == 0:
        return 0.0
    norm_a = math.sqrt(sum(count * count for count in a.values()))
    norm_b = math.sqrt(sum(count * count for count in b.values()))
    return dot / (norm_a * norm_b)


def _is_registry_conflict(matched_concept: Optional[StructuredConcept], candidate: Evidence) -> bool:
    """Phase 11B-5: prevents the exact false-match the diagnosis found —
    query "hemoglobin A1c" (which deterministically matches the registered
    "hba1c" concept, see structured_retriever.py) semantically overlapping
    with a plain Hemoglobin observation purely because both mention the
    word "hemoglobin". Bag-of-words cosine similarity alone cannot tell
    these apart; this adds one more DETERMINISTIC check on top of it,
    reusing the same curated concept registry structured retrieval already
    trusts: if the query matched a specific registered concept, and this
    candidate's own `code` is ITSELF a different registered concept's code
    (i.e. we have positive, curated knowledge — not a guess — that this
    candidate represents a distinct clinical concept), the candidate is
    rejected regardless of any lexical overlap.

    Deliberately narrow: this never fires for a query that doesn't match
    any registered concept (the common case — free-text queries like
    "blood pressure reading" are completely unaffected), and never fires
    against evidence whose code isn't itself a registered concept (so it
    can only ever reject a candidate we have explicit, curated grounds to
    consider a different, specific concept — never an arbitrary/unknown
    one)."""
    if matched_concept is None or candidate.code is None:
        return False
    candidate_concept_key = CODE_TO_CONCEPT_KEY.get(candidate.code)
    return candidate_concept_key is not None and candidate_concept_key != matched_concept.key


class SemanticEvidenceRetriever:
    """Deterministic, dependency-free semantic-ish EvidenceRetriever.
    Satisfies app.services.retrieval.EvidenceRetriever by structural
    typing (`retrieve(RetrievalRequest) -> RetrievalResult`; `limit` is an
    additional optional keyword argument)."""

    def __init__(self, db: Database, ranker: Optional[DeterministicEvidenceRanker] = None):
        self._evidence_service = EvidenceService(db)
        self._ranker = ranker or DeterministicEvidenceRanker()

    def retrieve(self, request: RetrievalRequest, *, limit: int = DEFAULT_LIMIT) -> RetrievalResult:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")

        query_vector = _tokenize(request.query)
        if not query_vector:
            return RetrievalResult.unsupported(
                patient_id=request.patient_id,
                query=request.query,
                message="Query contained no meaningful terms to search on",
            )

        # Phase 11B-5: if the query deterministically matches a specific
        # registered concept (e.g. "hba1c"), candidates whose own code is a
        # DIFFERENT registered concept are excluded below, regardless of
        # lexical overlap — see _is_registry_conflict.
        matched_concept = _match_concept(request.query)

        # Patient-scoped source query — this patient's Evidence only. There
        # is no multi-patient pool searched and filtered afterward.
        candidates = list(self._evidence_service.get_patient_evidence(request.patient_id))

        scored: list[tuple[float, Evidence]] = [
            (similarity, item)
            for item in candidates
            if not _is_registry_conflict(matched_concept, item)
            and (similarity := _cosine_similarity(query_vector, _tokenize(_evidence_text(item))))
            >= MIN_RELEVANCE_THRESHOLD
        ]

        if not scored:
            return RetrievalResult.from_evidence(patient_id=request.patient_id, query=request.query, evidence=[])

        # 1. similarity descending; 2-4. for exact ties, reuse
        # DeterministicEvidenceRanker's own deterministic ordering (which
        # ends in a stable resource_type/resource_id tie-break).
        by_similarity: dict[float, list[Evidence]] = {}
        for similarity, item in scored:
            by_similarity.setdefault(similarity, []).append(item)

        ordered: list[Evidence] = []
        for similarity in sorted(by_similarity, reverse=True):
            tied_items = by_similarity[similarity]
            ordered.extend(self._ranker.rank(tied_items, limit=len(tied_items)))

        return RetrievalResult.from_evidence(
            patient_id=request.patient_id, query=request.query, evidence=ordered[:limit]
        )
