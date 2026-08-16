"""The first real EvidenceRetriever (Phase 5A protocol) implementation:
deterministic structured retrieval.

Query interpretation and evidence retrieval are two separate, deterministic
steps:

  1. `_match_concept` maps free-text query to one of a small, fixed set of
     known structured clinical concepts via exact keyword/substring
     matching against a hardcoded registry — never fuzzy matching, never
     embeddings, never an LLM. A concept not in the registry is
     unsupported; nothing is guessed.
  2. `StructuredEvidenceRetriever.retrieve` calls the existing, already
     patient-scoped EvidenceService for that concept's FHIR code and wraps
     whatever real Evidence comes back (possibly none) into a
     RetrievalResult — it never fabricates or interprets.

NO EVIDENCE != NEGATIVE FACT: a supported query with nothing found returns
status="no_evidence_found", never a claim that the patient lacks the
condition/value. An unsupported query returns status="unsupported" rather
than guessing at a concept.

Every registry entry's code is verified present in this project's actual
ingested Synthea dataset (see the Phase 5B verification report) except
type_2_diabetes, which is a standard, correct SNOMED code that may or may
not have a match for any given patient — exactly like any other concept,
absence there is legitimately "no_evidence_found", not a registry defect.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pymongo.database import Database

from app.models.retrieval import RetrievalRequest, RetrievalResult
from app.services.evidence_ranker import DEFAULT_LIMIT, DeterministicEvidenceRanker
from app.services.evidence_service import EvidenceService


@dataclass(frozen=True)
class StructuredConcept:
    key: str
    resource_type: str
    code: str
    coding_system: str
    display: str
    # Lowercase substrings that identify this concept in free text. Exact
    # substring containment only — no fuzzy/edit-distance matching.
    triggers: tuple[str, ...]


# Order matters: checked top-to-bottom, first match wins. More specific
# concepts (e.g. "hba1c") are listed before broader ones that could
# otherwise shadow them (e.g. bare "hemoglobin").
_CONCEPT_REGISTRY: tuple[StructuredConcept, ...] = (
    StructuredConcept(
        key="hba1c",
        resource_type="Observation",
        code="4548-4",
        coding_system="http://loinc.org",
        display="Hemoglobin A1c",
        triggers=("hba1c", "a1c", "hemoglobin a1c", "glycated hemoglobin"),
    ),
    StructuredConcept(
        key="hemoglobin",
        resource_type="Observation",
        code="718-7",
        coding_system="http://loinc.org",
        display="Hemoglobin",
        triggers=("hemoglobin",),
    ),
    StructuredConcept(
        key="glucose",
        resource_type="Observation",
        code="2339-0",
        coding_system="http://loinc.org",
        display="Glucose",
        triggers=("glucose", "blood sugar"),
    ),
    StructuredConcept(
        key="bmi",
        resource_type="Observation",
        code="39156-5",
        coding_system="http://loinc.org",
        display="Body Mass Index",
        triggers=("bmi", "body mass index"),
    ),
    StructuredConcept(
        key="hypertension",
        resource_type="Condition",
        code="59621000",
        coding_system="http://snomed.info/sct",
        display="Hypertension",
        triggers=("hypertension", "high blood pressure"),
    ),
    StructuredConcept(
        key="prediabetes",
        resource_type="Condition",
        code="15777000",
        coding_system="http://snomed.info/sct",
        display="Prediabetes",
        triggers=("prediabetes", "pre-diabetes"),
    ),
    StructuredConcept(
        key="type_2_diabetes",
        resource_type="Condition",
        code="44054006",
        coding_system="http://snomed.info/sct",
        display="Type 2 diabetes mellitus",
        triggers=("type 2 diabetes", "type ii diabetes", "t2dm"),
    ),
    StructuredConcept(
        key="anemia",
        resource_type="Condition",
        code="271737000",
        coding_system="http://snomed.info/sct",
        display="Anemia",
        triggers=("anemia", "anaemia"),
    ),
    StructuredConcept(
        key="warfarin",
        resource_type="MedicationRequest",
        code="855332",
        coding_system="http://www.nlm.nih.gov/research/umls/rxnorm",
        display="Warfarin",
        triggers=("warfarin",),
    ),
    StructuredConcept(
        key="penicillin",
        resource_type="AllergyIntolerance",
        code="7984",
        coding_system="http://www.nlm.nih.gov/research/umls/rxnorm",
        display="Penicillin",
        triggers=("penicillin",),
    ),
)


def _match_concept(query: str) -> StructuredConcept | None:
    """Deterministic keyword lookup only. Returns None (unsupported) rather
    than guessing when nothing in the fixed registry matches."""
    normalized = query.strip().lower()
    for concept in _CONCEPT_REGISTRY:
        if any(trigger in normalized for trigger in concept.triggers):
            return concept
    return None


# Reverse lookup: a registered concept's own code -> its concept key. Used
# by app/services/semantic_retriever.py (Phase 11B-5) to deterministically
# recognize when a candidate's code is KNOWN, via this same curated
# registry, to represent a DIFFERENT specific clinical concept than the one
# the query matched (e.g. plain Hemoglobin, 718-7, is a different
# registered concept from HbA1c, 4548-4) — never a fuzzy/similarity-based
# guess, just a fact this registry already encodes.
CODE_TO_CONCEPT_KEY: dict[str, str] = {concept.code: concept.key for concept in _CONCEPT_REGISTRY}


@dataclass(frozen=True)
class CategoryConcept:
    """Phase 11B-2: a deterministic, explicitly allowlisted mapping from a
    CATEGORY of clinical data (e.g. "medications") to the FHIR resource_type
    that represents it — distinct from StructuredConcept above, which maps
    to one SPECIFIC named concept with one specific code. Never a guess:
    only the resource types listed here are ever reachable, and only via
    one of their own explicitly listed trigger words."""

    key: str
    resource_type: str
    triggers: tuple[str, ...]


# Deliberately small and explicit — do NOT turn arbitrary nouns into
# resource types. Each entry is a category of clinical data a patient's
# record may hold, mapped to the one FHIR resource_type that represents it.
_CATEGORY_REGISTRY: tuple[CategoryConcept, ...] = (
    CategoryConcept(
        key="medication_category",
        resource_type="MedicationRequest",
        triggers=("medications", "medication", "drugs", "drug"),
    ),
    CategoryConcept(
        key="condition_category",
        resource_type="Condition",
        triggers=("conditions", "condition", "diagnoses", "diagnosis"),
    ),
    CategoryConcept(
        key="allergy_category",
        resource_type="AllergyIntolerance",
        triggers=("allergies", "allergy"),
    ),
)


def _match_category(query: str) -> CategoryConcept | None:
    """Deterministic keyword lookup only, mirroring _match_concept exactly
    — returns None (unsupported) rather than guessing when nothing in the
    fixed category registry matches."""
    normalized = query.strip().lower()
    for category in _CATEGORY_REGISTRY:
        if any(trigger in normalized for trigger in category.triggers):
            return category
    return None


class StructuredEvidenceRetriever:
    """Deterministic structured EvidenceRetriever — satisfies the Phase 5A
    EvidenceRetriever protocol (app/services/retrieval.py) by structural
    typing (`retrieve(RetrievalRequest) -> RetrievalResult`).

    Phase 11B-2 adds a second, still fully deterministic lookup path:
    specific-concept matching (_match_concept, unchanged) is tried first;
    only if nothing specific matches does category matching
    (_match_category) get a chance — so an already-working specific query
    (e.g. "warfarin") is never affected by the new category fallback."""

    def __init__(self, db: Database, ranker: Optional[DeterministicEvidenceRanker] = None):
        self._evidence_service = EvidenceService(db)
        self._ranker = ranker or DeterministicEvidenceRanker()

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        concept = _match_concept(request.query)
        if concept is not None:
            # Patient-scoped, code-scoped lookup — never all-patients, never
            # unfiltered. The underlying generator is safely consumed in
            # full here because the query is already bounded to one
            # patient's resources of one code.
            evidence = list(
                self._evidence_service.get_patient_evidence_by_code(
                    request.patient_id, concept.code, resource_type=concept.resource_type
                )
            )
            return RetrievalResult.from_evidence(patient_id=request.patient_id, query=request.query, evidence=evidence)

        category = _match_category(request.query)
        if category is not None:
            # Patient-scoped, resource-type-scoped lookup — a category
            # question ("what medications...") maps to one resource_type,
            # never to another patient's data and never to an unbounded,
            # untyped scan. Ranked and truncated the same deterministic way
            # SemanticEvidenceRetriever already ranks its own candidates.
            candidates = list(
                self._evidence_service.get_patient_evidence(request.patient_id, resource_type=category.resource_type)
            )
            ranked = self._ranker.rank(
                candidates, patient_id=request.patient_id, resource_type_hint=category.resource_type, limit=DEFAULT_LIMIT
            )
            return RetrievalResult.from_evidence(patient_id=request.patient_id, query=request.query, evidence=ranked)

        return RetrievalResult.unsupported(
            patient_id=request.patient_id,
            query=request.query,
            message="No supported structured clinical concept could be deterministically identified in this query",
        )
