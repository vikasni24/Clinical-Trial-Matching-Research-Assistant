"""Phase 11B: retrieval & evidence quality fixes, directly targeting the
Phase 11A diagnosis — FHIR component[] values silently dropped, category
("What medications...") questions unsupported by structured retrieval,
plural-form vocabulary mismatch in the semantic retriever, and a
registry-known false semantic match (query "hemoglobin A1c" incorrectly
satisfied by a plain Hemoglobin observation). No real LLM is ever called;
LLM_API_KEY remains untouched throughout."""

from app.models.retrieval import RetrievalRequest
from app.repositories import evidence_repository
from app.services.evidence_service import EvidenceService
from app.services.hybrid_retriever import HybridEvidenceRetriever
from app.services.semantic_retriever import SemanticEvidenceRetriever, _normalize_token, _tokenize
from app.services.structured_retriever import StructuredEvidenceRetriever


def _insert_bp_observation(mongo_db, patient_id, resource_id, systolic, diastolic, effective_date="2023-01-10"):
    """A real FHIR panel-style Observation (LOINC 85354-9), matching the
    exact shape confirmed against the live dataset in Phase 11A — value
    lives in component[].valueQuantity, not a top-level value[x]."""
    mongo_db["fhir_resources"].insert_one(
        {
            "patient_id": patient_id,
            "resource_type": "Observation",
            "resource_id": resource_id,
            "data": {
                "resourceType": "Observation",
                "id": resource_id,
                "status": "final",
                "code": {"coding": [{"system": "http://loinc.org", "code": "85354-9", "display": "Blood Pressure"}], "text": "Blood Pressure"},
                "effectiveDateTime": effective_date,
                "component": [
                    {
                        "code": {"coding": [{"system": "http://loinc.org", "code": "8462-4", "display": "Diastolic Blood Pressure"}]},
                        "valueQuantity": {"value": diastolic, "unit": "mm[Hg]"},
                    },
                    {
                        "code": {"coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic Blood Pressure"}]},
                        "valueQuantity": {"value": systolic, "unit": "mm[Hg]"},
                    },
                ],
            },
        }
    )


def _insert_medication(mongo_db, patient_id, resource_id, code, display, status="active", authored_on="2023-01-01"):
    mongo_db["fhir_resources"].insert_one(
        {
            "patient_id": patient_id,
            "resource_type": "MedicationRequest",
            "resource_id": resource_id,
            "data": {
                "resourceType": "MedicationRequest",
                "id": resource_id,
                "status": status,
                "intent": "order",
                "medicationCodeableConcept": {
                    "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": code, "display": display}],
                    "text": display,
                },
                "subject": {"reference": f"Patient/{patient_id}"},
                "authoredOn": authored_on,
            },
        }
    )


def _insert_condition(mongo_db, patient_id, resource_id, code, display):
    mongo_db["fhir_resources"].insert_one(
        {
            "patient_id": patient_id,
            "resource_type": "Condition",
            "resource_id": resource_id,
            "data": {
                "resourceType": "Condition",
                "id": resource_id,
                "code": {"coding": [{"system": "http://snomed.info/sct", "code": code, "display": display}], "text": display},
                "clinicalStatus": {"coding": [{"code": "active"}]},
            },
        }
    )


def _insert_allergy(mongo_db, patient_id, resource_id, code, display):
    mongo_db["fhir_resources"].insert_one(
        {
            "patient_id": patient_id,
            "resource_type": "AllergyIntolerance",
            "resource_id": resource_id,
            "data": {
                "resourceType": "AllergyIntolerance",
                "id": resource_id,
                "code": {"coding": [{"system": "http://snomed.info/sct", "code": code, "display": display}], "text": display},
                "clinicalStatus": {"coding": [{"code": "active"}]},
            },
        }
    )


def _insert_observation(mongo_db, patient_id, resource_id, code, display, value, unit, effective_date="2023-01-10"):
    mongo_db["fhir_resources"].insert_one(
        {
            "patient_id": patient_id,
            "resource_type": "Observation",
            "resource_id": resource_id,
            "data": {
                "resourceType": "Observation",
                "id": resource_id,
                "status": "final",
                "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}], "text": display},
                "effectiveDateTime": effective_date,
                "valueQuantity": {"value": value, "unit": unit},
            },
        }
    )


# =====================================================================
# 11B-1 — FHIR component[] value extraction
# =====================================================================


def test_blood_pressure_component_values_are_extracted(mongo_db):
    _insert_bp_observation(mongo_db, "p1", "bp-1", systolic=125, diastolic=85)

    evidence = list(evidence_repository.get_patient_evidence(mongo_db, "p1", resource_type="Observation"))

    assert len(evidence) == 1
    assert evidence[0].value is not None
    assert "125" in evidence[0].value
    assert "85" in evidence[0].value


def test_systolic_and_diastolic_remain_distinguishable(mongo_db):
    _insert_bp_observation(mongo_db, "p1", "bp-1", systolic=140, diastolic=90)

    evidence = list(evidence_repository.get_patient_evidence(mongo_db, "p1", resource_type="Observation"))

    value_text = evidence[0].value.lower()
    assert "systolic" in value_text
    assert "diastolic" in value_text
    # The two numbers must not be merged/ambiguous — each is paired with
    # its own label in its own segment of the summary.
    segments = [segment.strip() for segment in value_text.split(";")]
    systolic_segment = next(s for s in segments if "systolic" in s)
    diastolic_segment = next(s for s in segments if "diastolic" in s)
    assert "140" in systolic_segment
    assert "90" not in systolic_segment
    assert "90" in diastolic_segment
    assert "140" not in diastolic_segment


def test_component_extraction_does_not_override_a_real_top_level_value(mongo_db):
    # A normal, non-panel Observation with its own top-level valueQuantity
    # must be completely unaffected by the new component-extraction path.
    _insert_observation(mongo_db, "p1", "obs-glucose", "2339-0", "Glucose", 95, "mg/dL")

    evidence = list(evidence_repository.get_patient_evidence(mongo_db, "p1", resource_type="Observation"))

    assert evidence[0].value == 95
    assert evidence[0].unit == "mg/dL"


def test_no_component_and_no_top_level_value_stays_none(mongo_db):
    # An Observation with neither a top-level value[x] nor a component[]
    # array must still, correctly, leave value as None — never fabricated.
    mongo_db["fhir_resources"].insert_one(
        {
            "patient_id": "p1",
            "resource_type": "Observation",
            "resource_id": "obs-no-value",
            "data": {
                "resourceType": "Observation",
                "id": "obs-no-value",
                "status": "final",
                "code": {"coding": [{"system": "http://loinc.org", "code": "1234-5", "display": "Something"}]},
            },
        }
    )

    evidence = list(evidence_repository.get_patient_evidence(mongo_db, "p1", resource_type="Observation"))

    assert evidence[0].value is None


def test_component_extraction_never_exposes_raw_fhir(mongo_db):
    _insert_bp_observation(mongo_db, "p1", "bp-1", systolic=125, diastolic=85)

    evidence = list(evidence_repository.get_patient_evidence(mongo_db, "p1", resource_type="Observation"))
    dumped = evidence[0].model_dump()

    assert "component" not in dumped
    assert "resourceType" not in dumped
    assert "data" not in dumped
    assert set(dumped.keys()) == set(type(evidence[0]).model_fields.keys())


# =====================================================================
# 11B-2 — category-aware structured retrieval
# =====================================================================


def test_medication_category_query_finds_real_medication(mongo_db):
    _insert_medication(mongo_db, "p1", "med-1", "313782", "Acetaminophen 325 MG Oral Tablet", status="stopped")
    retriever = StructuredEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="What medications is the patient taking?"))

    assert result.status == "evidence_found"
    assert result.evidence[0].resource_id == "med-1"
    assert result.evidence[0].display == "Acetaminophen 325 MG Oral Tablet"
    assert result.evidence[0].status == "stopped"


def test_condition_category_query_finds_real_conditions(mongo_db):
    _insert_condition(mongo_db, "p1", "cond-1", "10509002", "Acute bronchitis")
    retriever = StructuredEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="What conditions does the patient have?"))

    assert result.status == "evidence_found"
    assert result.evidence[0].resource_id == "cond-1"


def test_allergy_category_query_finds_real_allergies(mongo_db):
    _insert_allergy(mongo_db, "p1", "allergy-1", "91935009", "Peanut allergy")
    retriever = StructuredEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="What allergies are recorded?"))

    assert result.status == "evidence_found"
    assert result.evidence[0].resource_id == "allergy-1"


def test_category_query_with_no_matching_evidence_is_no_evidence_found_not_a_negative_claim(mongo_db):
    # Patient exists (has other resources) but genuinely no medications.
    _insert_condition(mongo_db, "p1", "cond-1", "10509002", "Acute bronchitis")
    retriever = StructuredEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="What medications is the patient taking?"))

    assert result.status == "no_evidence_found"
    assert result.evidence == []


def test_specific_concept_matching_still_takes_priority_over_category(mongo_db):
    # Warfarin is BOTH a specific registered concept AND would otherwise be
    # a MedicationRequest-category candidate — the specific, code-scoped
    # lookup must still be used (unchanged from before Phase 11B).
    _insert_medication(mongo_db, "p1", "med-warfarin", "855332", "Warfarin 5 MG Oral Tablet")
    _insert_medication(mongo_db, "p1", "med-other", "313782", "Acetaminophen 325 MG Oral Tablet")
    retriever = StructuredEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="warfarin"))

    assert result.status == "evidence_found"
    assert [e.resource_id for e in result.evidence] == ["med-warfarin"]


def test_category_retrieval_is_ranked_and_bounded(mongo_db):
    for i in range(8):
        _insert_medication(mongo_db, "p1", f"med-{i}", "313782", f"Medication {i}")
    retriever = StructuredEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="medications"))

    assert result.status == "evidence_found"
    assert len(result.evidence) <= 5  # DeterministicEvidenceRanker's own DEFAULT_LIMIT


def test_category_retrieval_never_creates_a_multi_patient_pool(mongo_db, monkeypatch):
    _insert_medication(mongo_db, "p1", "med-1", "313782", "Acetaminophen")
    collection = mongo_db["fhir_resources"]
    original_find = collection.find
    captured = []

    def spying_find(filter=None, *args, **kwargs):
        captured.append(filter or {})
        return original_find(filter, *args, **kwargs)

    monkeypatch.setattr(collection, "find", spying_find)

    StructuredEvidenceRetriever(mongo_db).retrieve(RetrievalRequest(patient_id="p1", query="medications"))

    assert captured
    for query in captured:
        assert query != {}
        assert query.get("patient_id") == "p1"


def test_category_retrieval_does_not_bypass_hybrid_retriever(mongo_db):
    _insert_medication(mongo_db, "p1", "med-1", "313782", "Acetaminophen 325 MG Oral Tablet")
    hybrid = HybridEvidenceRetriever(mongo_db)

    result = hybrid.retrieve(RetrievalRequest(patient_id="p1", query="What medications is the patient taking?"))

    assert result.status == "evidence_found"
    assert any(e.resource_id == "med-1" for e in result.evidence)


# =====================================================================
# 11B-3 — plural query normalization
# =====================================================================


def test_normalize_token_medication_forms():
    assert _normalize_token("medication") == "medication"
    assert _normalize_token("medications") == "medication"


def test_normalize_token_condition_forms():
    assert _normalize_token("condition") == "condition"
    assert _normalize_token("conditions") == "condition"


def test_normalize_token_allergy_forms():
    assert _normalize_token("allergy") == "allergy"
    assert _normalize_token("allergies") == "allergy"


def test_normalize_token_drug_forms():
    assert _normalize_token("drug") == "drug"
    assert _normalize_token("drugs") == "drug"


def test_normalize_token_does_not_damage_unrelated_words():
    # None of these are in the explicit allowlist, so a naive "strip
    # trailing s" rule would mangle them — the actual implementation must
    # leave every one of them untouched.
    assert _normalize_token("status") == "status"
    assert _normalize_token("diabetes") == "diabetes"
    assert _normalize_token("blood") == "blood"
    assert _normalize_token("pressure") == "pressure"
    assert _normalize_token("hypertension") == "hypertension"
    assert _normalize_token("hemoglobin") == "hemoglobin"


def test_tokenize_applies_normalization_consistently():
    # Query and evidence text must normalize the same way for bag-of-words
    # comparison to work symmetrically.
    assert _tokenize("medications") == _tokenize("medication")
    assert "allergy" in _tokenize("What allergies are recorded?")
    assert "allergies" not in _tokenize("What allergies are recorded?")


def test_singular_and_plural_medication_query_both_find_the_same_evidence(mongo_db):
    _insert_medication(mongo_db, "p1", "med-1", "313782", "Acetaminophen 325 MG Oral Tablet")
    retriever = SemanticEvidenceRetriever(mongo_db)

    singular = retriever.retrieve(RetrievalRequest(patient_id="p1", query="medication"))
    plural = retriever.retrieve(RetrievalRequest(patient_id="p1", query="medications"))

    # Both should behave identically now (previously "medications" alone
    # found nothing at all — see Phase 11A diagnosis).
    assert singular.status == plural.status


# =====================================================================
# 11B-4 — category questions reflect evidence status, never overstate it
# =====================================================================


def test_stopped_medication_status_is_preserved_not_dropped(mongo_db):
    _insert_medication(mongo_db, "p1", "med-1", "313782", "Acetaminophen 325 MG Oral Tablet", status="stopped")
    retriever = StructuredEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="What medications is the patient taking?"))

    assert result.evidence[0].status == "stopped"


def test_prompt_surfaces_medication_status_for_the_llm_to_phrase_accurately(mongo_db):
    from app.services.grounded_prompt import build_grounded_prompt
    from app.services.rag_context import build_grounded_context

    _insert_medication(mongo_db, "p1", "med-1", "313782", "Acetaminophen 325 MG Oral Tablet", status="stopped")
    result = StructuredEvidenceRetriever(mongo_db).retrieve(
        RetrievalRequest(patient_id="p1", query="What medications is the patient taking?")
    )
    context = build_grounded_context(result)
    prompt = build_grounded_prompt(context)

    # The status is visible in the rendered evidence text the LLM receives
    # — the deterministic layer never silently drops it, which is what
    # would let an LLM claim "currently taking" without contradicting
    # evidence. Whether the LLM actually phrases it correctly is outside
    # what deterministic code can guarantee — but it can guarantee the LLM
    # was never denied the information needed to do so.
    assert "stopped" in prompt.evidence_text.lower()
    assert "Acetaminophen" in prompt.evidence_text


def test_no_condition_evidence_is_not_a_negative_claim_end_to_end(mongo_db):
    _insert_medication(mongo_db, "p1", "med-1", "313782", "Acetaminophen 325 MG Oral Tablet")
    hybrid = HybridEvidenceRetriever(mongo_db)

    result = hybrid.retrieve(RetrievalRequest(patient_id="p1", query="What conditions does the patient have?"))

    assert result.status == "no_evidence_found"
    if result.message:
        assert "no conditions" not in result.message.lower()
        assert "does not have" not in result.message.lower()


# =====================================================================
# 11B-5 — prevent false semantic matches on generic vocabulary overlap
# =====================================================================


def test_hemoglobin_a1c_query_does_not_return_plain_hemoglobin(mongo_db):
    _insert_observation(mongo_db, "p1", "obs-hgb", "718-7", "Hemoglobin [Mass/volume] in Blood", 14.5, "g/dL")
    retriever = SemanticEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="hemoglobin A1c"))

    assert result.status == "no_evidence_found"
    assert result.evidence == []


def test_hba1c_query_does_not_return_plain_hemoglobin_via_hybrid(mongo_db):
    _insert_observation(mongo_db, "p1", "obs-hgb", "718-7", "Hemoglobin [Mass/volume] in Blood", 14.5, "g/dL")
    hybrid = HybridEvidenceRetriever(mongo_db)

    result = hybrid.retrieve(RetrievalRequest(patient_id="p1", query="What is the patient's HbA1c?"))

    assert result.status == "no_evidence_found"
    assert result.evidence == []


def test_real_hba1c_evidence_is_still_found_normally(mongo_db):
    # The registry-conflict filter must never reject genuinely correct
    # evidence for the concept that was actually matched.
    _insert_observation(mongo_db, "p1", "obs-hba1c", "4548-4", "Hemoglobin A1c", 5.81, "%")
    retriever = SemanticEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="hemoglobin A1c"))

    assert result.status == "evidence_found"
    assert result.evidence[0].resource_id == "obs-hba1c"


def test_hypertension_query_does_not_match_a_different_registered_concept_via_word_overlap(mongo_db):
    # Anemia (a different registered concept, code 271737000) whose display
    # text happens to contain the word "hypertension" — the registry
    # already knows this code belongs to a different concept, so the
    # conflict filter must reject it even though the word literally
    # overlaps.
    _insert_condition(mongo_db, "p1", "cond-other", "271737000", "Hypertension-related anemia (unrelated concept)")
    retriever = SemanticEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="hypertension"))

    assert result.status == "no_evidence_found"
    assert result.evidence == []


def test_registry_conflict_filter_does_not_affect_queries_matching_no_concept(mongo_db):
    # "blood pressure reading" doesn't match any specific registered
    # concept, so the new filter must be a complete no-op here — existing
    # behavior (test_relevant_evidence_ranks_above_unrelated_evidence in
    # test_semantic_retriever.py) is unaffected.
    _insert_observation(mongo_db, "p1", "obs-bp", "8480-6", "Systolic Blood Pressure", 125, "mm[Hg]")
    retriever = SemanticEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="blood pressure reading"))

    assert result.status == "evidence_found"
    assert result.evidence[0].resource_id == "obs-bp"


# =====================================================================
# 11B-6 — patient isolation for category queries
# =====================================================================


def test_medication_category_query_never_crosses_patients(mongo_db):
    _insert_medication(mongo_db, "patient-a", "med-a", "313782", "Acetaminophen 325 MG Oral Tablet")
    _insert_medication(mongo_db, "patient-b", "med-b", "860975", "Metformin 500 MG Oral Tablet")

    structured = StructuredEvidenceRetriever(mongo_db)
    result_a = structured.retrieve(RetrievalRequest(patient_id="patient-a", query="medications"))
    result_b = structured.retrieve(RetrievalRequest(patient_id="patient-b", query="medications"))

    assert [e.resource_id for e in result_a.evidence] == ["med-a"]
    assert [e.resource_id for e in result_b.evidence] == ["med-b"]
    assert all(e.patient_id == "patient-a" for e in result_a.evidence)
    assert all(e.patient_id == "patient-b" for e in result_b.evidence)


def test_medication_category_query_never_crosses_patients_via_hybrid(mongo_db):
    _insert_medication(mongo_db, "patient-a", "med-a", "313782", "Acetaminophen 325 MG Oral Tablet")
    _insert_medication(mongo_db, "patient-b", "med-b", "860975", "Metformin 500 MG Oral Tablet")

    hybrid = HybridEvidenceRetriever(mongo_db)
    result_a = hybrid.retrieve(RetrievalRequest(patient_id="patient-a", query="What medications is the patient taking?"))
    result_b = hybrid.retrieve(RetrievalRequest(patient_id="patient-b", query="What medications is the patient taking?"))

    ids_a = {e.resource_id for e in result_a.evidence}
    ids_b = {e.resource_id for e in result_b.evidence}
    assert "med-a" in ids_a and "med-b" not in ids_a
    assert "med-b" in ids_b and "med-a" not in ids_b
    assert all(e.patient_id == "patient-a" for e in result_a.evidence)
    assert all(e.patient_id == "patient-b" for e in result_b.evidence)


def test_patient_a_gets_no_evidence_when_only_patient_b_has_medications(mongo_db):
    _insert_medication(mongo_db, "patient-b", "med-b", "860975", "Metformin 500 MG Oral Tablet")
    structured = StructuredEvidenceRetriever(mongo_db)

    result_a = structured.retrieve(RetrievalRequest(patient_id="patient-a", query="medications"))

    assert result_a.status == "no_evidence_found"
    assert result_a.evidence == []


# =====================================================================
# 11B-7 — safety states preserved
# =====================================================================


def test_unsupported_query_still_returns_unsupported_after_category_addition(mongo_db):
    retriever = StructuredEvidenceRetriever(mongo_db)

    result = retriever.retrieve(RetrievalRequest(patient_id="p1", query="Tell me a joke"))

    assert result.status == "unsupported"
    assert result.evidence == []


def test_pre_generation_safety_gate_still_prevents_llm_call_for_no_evidence_category_query(mongo_db):
    from app.services.rag_context import GroundedContextService
    from app.services.safety_rules import enforce_pre_generation_safety

    context = GroundedContextService(mongo_db).build_context(
        RetrievalRequest(patient_id="p1", query="What medications is the patient taking?")
    )
    gated = enforce_pre_generation_safety(context)

    assert context.status == "no_evidence_found"
    assert gated is not None
    assert gated.status == "insufficient_evidence"


def test_fabricated_citation_still_rejected_for_category_evidence(mongo_db):
    from app.services.answer_validator import build_grounded_answer
    from app.services.rag_context import GroundedContextService

    _insert_medication(mongo_db, "p1", "med-1", "313782", "Acetaminophen 325 MG Oral Tablet")
    context = GroundedContextService(mongo_db).build_context(
        RetrievalRequest(patient_id="p1", query="What medications is the patient taking?")
    )
    assert context.status == "evidence_found"

    answer = build_grounded_answer(context, "The patient takes [MedicationRequest/does-not-exist].")

    assert answer.status == "insufficient_evidence"
    assert answer.evidence == []


def test_cross_patient_evidence_still_rejected_by_safety_gate(mongo_db):
    from app.models.evidence import Evidence
    from app.models.rag_context import GroundedContext
    from app.services.safety_rules import enforce_pre_generation_safety
    import pytest as _pytest

    contaminated = GroundedContext.model_construct(
        patient_id="patient-a",
        query="What medications is the patient taking?",
        status="evidence_found",
        evidence=[Evidence(patient_id="patient-b", resource_type="MedicationRequest", resource_id="med-b", display="Metformin")],
        message=None,
    )

    with _pytest.raises(ValueError):
        enforce_pre_generation_safety(contaminated)


# =====================================================================
# 11B-8 — realistic natural-language question matrix
# =====================================================================


def _setup_realistic_patient(mongo_db, patient_id="p1"):
    _insert_medication(mongo_db, patient_id, "med-1", "313782", "Acetaminophen 325 MG Oral Tablet", status="stopped")
    _insert_condition(mongo_db, patient_id, "cond-1", "10509002", "Acute bronchitis")
    _insert_allergy(mongo_db, patient_id, "allergy-1", "91935009", "Peanut allergy")
    _insert_bp_observation(mongo_db, patient_id, "bp-1", systolic=125, diastolic=85)


def test_what_medications_is_the_patient_taking(mongo_db):
    _setup_realistic_patient(mongo_db)
    result = HybridEvidenceRetriever(mongo_db).retrieve(
        RetrievalRequest(patient_id="p1", query="What medications is the patient taking?")
    )
    assert result.status == "evidence_found"
    assert any(e.resource_id == "med-1" for e in result.evidence)


def test_list_medications(mongo_db):
    _setup_realistic_patient(mongo_db)
    result = HybridEvidenceRetriever(mongo_db).retrieve(RetrievalRequest(patient_id="p1", query="List medications"))
    assert result.status == "evidence_found"
    assert any(e.resource_id == "med-1" for e in result.evidence)


def test_what_drugs_are_recorded(mongo_db):
    _setup_realistic_patient(mongo_db)
    result = HybridEvidenceRetriever(mongo_db).retrieve(RetrievalRequest(patient_id="p1", query="What drugs are recorded?"))
    assert result.status == "evidence_found"
    assert any(e.resource_id == "med-1" for e in result.evidence)


def test_what_conditions_does_the_patient_have(mongo_db):
    _setup_realistic_patient(mongo_db)
    result = HybridEvidenceRetriever(mongo_db).retrieve(
        RetrievalRequest(patient_id="p1", query="What conditions does the patient have?")
    )
    assert result.status == "evidence_found"
    assert any(e.resource_id == "cond-1" for e in result.evidence)


def test_list_conditions(mongo_db):
    _setup_realistic_patient(mongo_db)
    result = HybridEvidenceRetriever(mongo_db).retrieve(RetrievalRequest(patient_id="p1", query="List conditions"))
    assert result.status == "evidence_found"
    assert any(e.resource_id == "cond-1" for e in result.evidence)


def test_what_allergies_are_recorded(mongo_db):
    _setup_realistic_patient(mongo_db)
    result = HybridEvidenceRetriever(mongo_db).retrieve(RetrievalRequest(patient_id="p1", query="What allergies are recorded?"))
    assert result.status == "evidence_found"
    assert any(e.resource_id == "allergy-1" for e in result.evidence)


def test_what_is_the_patients_blood_pressure(mongo_db):
    _setup_realistic_patient(mongo_db)
    result = HybridEvidenceRetriever(mongo_db).retrieve(
        RetrievalRequest(patient_id="p1", query="What is the patient's blood pressure?")
    )
    assert result.status == "evidence_found"
    assert any(e.resource_id == "bp-1" for e in result.evidence)
    matched = next(e for e in result.evidence if e.resource_id == "bp-1")
    assert "125" in matched.value and "85" in matched.value


def test_what_is_the_systolic_blood_pressure(mongo_db):
    _setup_realistic_patient(mongo_db)
    result = HybridEvidenceRetriever(mongo_db).retrieve(
        RetrievalRequest(patient_id="p1", query="What is the systolic blood pressure?")
    )
    assert result.status == "evidence_found"
    assert any(e.resource_id == "bp-1" for e in result.evidence)


def test_what_is_the_diastolic_blood_pressure(mongo_db):
    _setup_realistic_patient(mongo_db)
    result = HybridEvidenceRetriever(mongo_db).retrieve(
        RetrievalRequest(patient_id="p1", query="What is the diastolic blood pressure?")
    )
    assert result.status == "evidence_found"
    assert any(e.resource_id == "bp-1" for e in result.evidence)


def test_hba1c_query_for_patient_without_hba1c_evidence(mongo_db):
    # Patient has real evidence (medications, conditions, BP, plain
    # Hemoglobin) but genuinely no HbA1c — must remain no_evidence_found,
    # and must NOT fall back to the unrelated plain Hemoglobin observation.
    _setup_realistic_patient(mongo_db)
    _insert_observation(mongo_db, "p1", "obs-hgb", "718-7", "Hemoglobin [Mass/volume] in Blood", 14.5, "g/dL")

    result = HybridEvidenceRetriever(mongo_db).retrieve(RetrievalRequest(patient_id="p1", query="What is the patient's HbA1c?"))

    assert result.status == "no_evidence_found"
    assert result.evidence == []


def test_hba1c_no_evidence_becomes_insufficient_evidence_not_llm_call(mongo_db):
    from app.services.rag_context import GroundedContextService
    from app.services.safety_rules import enforce_pre_generation_safety

    _setup_realistic_patient(mongo_db)
    _insert_observation(mongo_db, "p1", "obs-hgb", "718-7", "Hemoglobin [Mass/volume] in Blood", 14.5, "g/dL")

    context = GroundedContextService(mongo_db).build_context(
        RetrievalRequest(patient_id="p1", query="What is the patient's HbA1c?")
    )
    gated = enforce_pre_generation_safety(context)

    assert context.status == "no_evidence_found"
    assert gated is not None
    assert gated.status == "insufficient_evidence"
    assert gated.answer_text is None
