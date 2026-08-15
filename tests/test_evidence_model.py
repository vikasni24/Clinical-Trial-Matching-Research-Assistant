import pytest
from pydantic import ValidationError

from app.models.evidence import Evidence


def test_valid_observation_evidence():
    evidence = Evidence(
        patient_id="p1",
        resource_type="Observation",
        resource_id="obs-1",
        code="4548-4",
        coding_system="http://loinc.org",
        display="Hemoglobin A1c",
        value=5.81,
        unit="%",
        effective_date="2023-01-10",
        status="final",
    )

    assert evidence.resource_type == "Observation"
    assert evidence.value == 5.81
    assert evidence.unit == "%"
    assert evidence.code == "4548-4"


def test_valid_condition_evidence():
    evidence = Evidence(
        patient_id="p1",
        resource_type="Condition",
        resource_id="cond-1",
        code="38341003",
        coding_system="http://snomed.info/sct",
        display="Hypertension",
        status="active",
    )

    assert evidence.resource_type == "Condition"
    assert evidence.display == "Hypertension"
    assert evidence.value is None  # Conditions don't carry a numeric value


def test_valid_medication_evidence():
    evidence = Evidence(
        patient_id="p1",
        resource_type="MedicationRequest",
        resource_id="med-1",
        code="860975",
        coding_system="http://www.nlm.nih.gov/research/umls/rxnorm",
        display="Metformin",
        status="active",
    )

    assert evidence.resource_type == "MedicationRequest"
    assert evidence.code == "860975"


def test_valid_allergy_evidence():
    evidence = Evidence(
        patient_id="p1",
        resource_type="AllergyIntolerance",
        resource_id="allergy-1",
        code="91935009",
        coding_system="http://snomed.info/sct",
        display="Peanut",
        status="active",
    )

    assert evidence.resource_type == "AllergyIntolerance"
    assert evidence.display == "Peanut"


def test_optional_fields_can_be_absent():
    # Only the required traceability fields are supplied — every clinical
    # content field is legitimately absent, as for a Patient demographics record.
    evidence = Evidence(patient_id="p1", resource_type="Patient", resource_id="p1")

    assert evidence.code is None
    assert evidence.coding_system is None
    assert evidence.display is None
    assert evidence.value is None
    assert evidence.unit is None
    assert evidence.effective_date is None
    assert evidence.status is None


def test_empty_patient_id_rejected():
    with pytest.raises(ValidationError):
        Evidence(patient_id="", resource_type="Observation", resource_id="obs-1")


def test_empty_resource_type_rejected():
    with pytest.raises(ValidationError):
        Evidence(patient_id="p1", resource_type="", resource_id="obs-1")


def test_empty_resource_id_rejected():
    with pytest.raises(ValidationError):
        Evidence(patient_id="p1", resource_type="Observation", resource_id="")


def test_resource_id_is_preserved_exactly():
    evidence = Evidence(patient_id="p1", resource_type="Observation", resource_id="obs-abc-123-XYZ")

    assert evidence.resource_id == "obs-abc-123-XYZ"


def test_patient_id_is_preserved_exactly():
    evidence = Evidence(
        patient_id="b0a06ead-cc42-aa48-dad6-841d4aa679fa", resource_type="Observation", resource_id="obs-1"
    )

    assert evidence.patient_id == "b0a06ead-cc42-aa48-dad6-841d4aa679fa"


def test_evidence_does_not_invent_missing_values():
    # No value/unit/effective_date supplied for this Condition-type evidence —
    # the model must leave them unset, never substitute a default or guess.
    evidence = Evidence(patient_id="p1", resource_type="Condition", resource_id="cond-1", display="Hypertension")

    assert evidence.value is None
    assert evidence.unit is None
    assert evidence.effective_date is None
    assert evidence.code is None
    assert evidence.coding_system is None


def test_evidence_remains_serializable_through_pydantic():
    evidence = Evidence(
        patient_id="p1",
        resource_type="Observation",
        resource_id="obs-1",
        code="4548-4",
        coding_system="http://loinc.org",
        display="Hemoglobin A1c",
        value=5.81,
        unit="%",
        effective_date="2023-01-10",
    )

    dumped = evidence.model_dump()
    assert dumped["patient_id"] == "p1"
    assert dumped["value"] == 5.81

    json_text = evidence.model_dump_json()
    restored = Evidence.model_validate_json(json_text)
    assert restored == evidence


def test_source_collection_defaults_to_fhir_resources():
    evidence = Evidence(patient_id="p1", resource_type="Observation", resource_id="obs-1")

    assert evidence.source_collection == "fhir_resources"


def test_source_reference_defaults_from_resource_type_and_id():
    evidence = Evidence(patient_id="p1", resource_type="Observation", resource_id="obs-1")

    assert evidence.source_reference == "Observation/obs-1"


def test_evidence_model_has_no_ai_generated_or_scoring_fields():
    # Clinical safety rule: the ground-truth evidence layer must never carry
    # fields that imply an AI-generated conclusion or confidence score.
    prohibited_fields = {
        "diagnosis_inference",
        "predicted_condition",
        "confidence_score",
        "ai_reasoning",
        "generated_evidence",
        "semantic_similarity",
        "hallucination_score",
    }
    assert prohibited_fields.isdisjoint(Evidence.model_fields.keys())
