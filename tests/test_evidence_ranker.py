"""Phase 5C: deterministic Evidence ranking/selection. Pure unit tests —
no MongoDB, no fixtures; the ranker operates only on already-constructed
Evidence objects."""

import inspect

import pytest

from app.models.evidence import Evidence
from app.services.evidence_ranker import DeterministicEvidenceRanker


def _evidence(resource_id="obs-1", patient_id="p1", effective_date=None, value=None, code="4548-4", resource_type="Observation"):
    return Evidence(
        patient_id=patient_id,
        resource_type=resource_type,
        resource_id=resource_id,
        code=code,
        value=value,
        effective_date=effective_date,
    )


ranker = DeterministicEvidenceRanker()


# --- 1-2: deterministic ordering, recency ---------------------------------------


def test_evidence_is_ranked_deterministically_regardless_of_input_order():
    a = _evidence("obs-a", effective_date="2020-01-01")
    b = _evidence("obs-b", effective_date="2021-01-01")
    c = _evidence("obs-c", effective_date="2019-01-01")

    result1 = ranker.rank([a, b, c], limit=10)
    result2 = ranker.rank([c, a, b], limit=10)
    result3 = ranker.rank([b, c, a], limit=10)

    ids1 = [e.resource_id for e in result1]
    assert ids1 == [e.resource_id for e in result2] == [e.resource_id for e in result3]


def test_more_recent_evidence_ranks_before_older_evidence():
    old = _evidence("obs-old", effective_date="2001-01-01")
    new = _evidence("obs-new", effective_date="2023-01-01")

    ranked = ranker.rank([old, new], limit=10)

    assert [e.resource_id for e in ranked] == ["obs-new", "obs-old"]


# --- 3-4: missing / malformed dates never crash ---------------------------------


def test_missing_dates_do_not_crash_and_sort_after_dated_evidence():
    dated = _evidence("obs-dated", effective_date="2020-01-01")
    undated = _evidence("obs-undated", effective_date=None)

    ranked = ranker.rank([undated, dated], limit=10)

    assert [e.resource_id for e in ranked] == ["obs-dated", "obs-undated"]


def test_malformed_dates_do_not_crash_and_sort_after_valid_dates():
    dated = _evidence("obs-dated", effective_date="2020-01-01")
    malformed = _evidence("obs-malformed", effective_date="definitely-not-a-date")

    ranked = ranker.rank([malformed, dated], limit=10)

    assert [e.resource_id for e in ranked] == ["obs-dated", "obs-malformed"]


def test_mixed_timezone_aware_and_naive_dates_do_not_crash():
    naive = _evidence("obs-naive", effective_date="2015-06-01")
    aware = _evidence("obs-aware", effective_date="2020-06-15T03:14:47-05:00")

    ranked = ranker.rank([naive, aware], limit=10)

    assert [e.resource_id for e in ranked] == ["obs-aware", "obs-naive"]


# --- 5: stable tie-breaking -------------------------------------------------------


def test_stable_tie_breaking_by_resource_id():
    # Identical relevance signals (same code presence, no value, no date) —
    # only resource_id differs, so alphabetical resource_id must decide.
    z = _evidence("obs-z")
    a = _evidence("obs-a")

    ranked = ranker.rank([z, a], limit=10)

    assert [e.resource_id for e in ranked] == ["obs-a", "obs-z"]


# --- 6-9: limit / top-K behavior --------------------------------------------------


def test_limit_1_returns_exactly_one_item():
    items = [_evidence(f"obs-{i}", effective_date=f"20{10+i:02d}-01-01") for i in range(5)]

    ranked = ranker.rank(items, limit=1)

    assert len(ranked) == 1


def test_limit_5_returns_at_most_five():
    items = [_evidence(f"obs-{i}", effective_date=f"20{10+i:02d}-01-01") for i in range(8)]

    ranked = ranker.rank(items, limit=5)

    assert len(ranked) == 5


def test_limit_larger_than_available_returns_all():
    items = [_evidence("obs-1"), _evidence("obs-2")]

    ranked = ranker.rank(items, limit=100)

    assert len(ranked) == 2


def test_nonpositive_limit_is_rejected():
    with pytest.raises(ValueError):
        ranker.rank([_evidence()], limit=0)

    with pytest.raises(ValueError):
        ranker.rank([_evidence()], limit=-1)


def test_default_limit_is_five():
    items = [_evidence(f"obs-{i}", effective_date=f"20{10+i:02d}-01-01") for i in range(8)]

    ranked = ranker.rank(items)

    assert len(ranked) == 5


# --- 10: empty evidence ------------------------------------------------------------


def test_empty_evidence_returns_empty_list():
    assert ranker.rank([], limit=5) == []


# --- 11-12: determinism and non-mutation -------------------------------------------


def test_repeated_ranking_of_identical_input_is_byte_identical():
    items = [_evidence("obs-a", effective_date="2020-01-01"), _evidence("obs-b", effective_date="2021-01-01")]

    first = ranker.rank(items, limit=10)
    second = ranker.rank(items, limit=10)

    assert [e.model_dump() for e in first] == [e.model_dump() for e in second]


def test_evidence_objects_remain_unchanged():
    original = _evidence("obs-1", effective_date="2020-01-01", value=5.81)

    ranked = ranker.rank([original], limit=10)

    assert ranked[0] == original


# --- 13: no raw FHIR introduced -----------------------------------------------------


def test_no_raw_fhir_document_introduced():
    ranked = ranker.rank([_evidence("obs-1", value=5.81)], limit=10)

    dumped = ranked[0].model_dump()
    assert "data" not in dumped
    assert "resourceType" not in dumped
    assert "_id" not in dumped


# --- 14: cross-patient evidence cannot be silently mixed ----------------------------


def test_cross_patient_evidence_is_rejected_when_patient_id_given():
    same_patient = _evidence("obs-1", patient_id="p1")
    other_patient = _evidence("obs-2", patient_id="p2")

    with pytest.raises(ValueError):
        ranker.rank([same_patient, other_patient], patient_id="p1", limit=10)


def test_ranking_without_patient_id_does_not_validate_isolation():
    # patient_id is optional — when omitted, the ranker trusts the caller
    # already scoped the list (documented behavior, not a defect).
    mixed = [_evidence("obs-1", patient_id="p1"), _evidence("obs-2", patient_id="p2")]

    ranked = ranker.rank(mixed, limit=10)  # no patient_id passed

    assert len(ranked) == 2


# --- 15: ranking score is relevance-only, never confidence -------------------------


def test_ranking_module_never_uses_clinical_confidence_terminology():
    # The module's docstrings legitimately use words like "confidence" and
    # "certainty" in negation prose ("never scored for medical confidence")
    # — so this checks for actual field/variable-style naming, not the bare
    # words alone.
    import app.services.evidence_ranker as module

    source = inspect.getsource(module).lower()
    for forbidden in ("confidence_score", "truth_probability", "clinical_certainty", "hallucination_score", "confidence ="):
        assert forbidden not in source


def test_ranked_evidence_has_no_added_score_field():
    ranked = ranker.rank([_evidence("obs-1", value=5.81)], limit=10)

    # The ranker returns plain Evidence objects — no wrapper, no score field added.
    assert set(ranked[0].model_dump().keys()) == set(Evidence.model_fields.keys())


# --- 16: Phase 5B's retrieval behavior is unchanged / composable -------------------


def test_composes_with_structured_retriever_output_without_modifying_it(mongo_db):
    from app.models.retrieval import RetrievalRequest
    from app.services.structured_retriever import StructuredEvidenceRetriever

    mongo_db["fhir_resources"].insert_many(
        [
            {
                "patient_id": "p1",
                "resource_type": "Observation",
                "resource_id": f"obs-{i}",
                "data": {
                    "resourceType": "Observation",
                    "id": f"obs-{i}",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "Hemoglobin A1c"}]},
                    "effectiveDateTime": f"20{10 + i:02d}-01-01",
                    "valueQuantity": {"value": 5.0 + i, "unit": "%"},
                },
            }
            for i in range(6)
        ]
    )

    retrieval_result = StructuredEvidenceRetriever(mongo_db).retrieve(
        RetrievalRequest(patient_id="p1", query="What is the patient's HbA1c?")
    )
    assert retrieval_result.status == "evidence_found"
    assert len(retrieval_result.evidence) == 6  # Phase 5B still returns full history, unchanged

    ranked = ranker.rank(retrieval_result.evidence, patient_id="p1", limit=3)

    assert len(ranked) == 3
    assert ranked[0].resource_id == "obs-5"  # most recent (2015) ranks first
