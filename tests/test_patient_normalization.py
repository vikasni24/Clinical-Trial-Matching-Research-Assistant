from app.services.patient_normalization import (
    build_patient_profile,
    normalize_allergy,
    normalize_condition,
    normalize_contact,
    normalize_demographics,
    normalize_diagnostic_report,
    normalize_encounter,
    normalize_medication,
    normalize_observation,
    normalize_procedure,
)


def _doc(resource_type: str, resource_id: str, patient_id: str, data: dict) -> dict:
    return {"resource_type": resource_type, "resource_id": resource_id, "patient_id": patient_id, "data": data}


# --- demographics -----------------------------------------------------------


def test_demographic_normalization():
    patient_data = {
        "id": "p1",
        "name": [{"family": "Doe", "given": ["Jane", "Q"]}],
        "gender": "female",
        "birthDate": "1980-05-12",
        "maritalStatus": {"text": "Married"},
        "deceasedDateTime": None,
        "extension": [
            {
                "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
                "extension": [{"url": "text", "valueString": "White"}],
            },
            {
                "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity",
                "extension": [{"url": "text", "valueString": "Not Hispanic or Latino"}],
            },
        ],
    }
    demographics = normalize_demographics(patient_data)

    assert demographics.patient_id == "p1"
    assert demographics.first_name == "Jane"
    assert demographics.last_name == "Doe"
    assert demographics.full_name == "Jane Q Doe"
    assert demographics.date_of_birth == "1980-05-12"
    assert demographics.gender == "female"
    assert demographics.race == "White"
    assert demographics.ethnicity == "Not Hispanic or Latino"
    assert demographics.marital_status == "Married"
    assert demographics.deceased is False
    assert demographics.deceased_date is None


def test_demographic_normalization_deceased_patient():
    patient_data = {"id": "p2", "deceasedDateTime": "2020-01-01"}
    demographics = normalize_demographics(patient_data)

    assert demographics.deceased is True
    assert demographics.deceased_date == "2020-01-01"


def test_demographic_normalization_missing_optional_fields():
    # Only the bare minimum FHIR Patient fields — everything else absent.
    patient_data = {"id": "p3"}
    demographics = normalize_demographics(patient_data)

    assert demographics.patient_id == "p3"
    assert demographics.first_name is None
    assert demographics.last_name is None
    assert demographics.full_name is None
    assert demographics.race is None
    assert demographics.ethnicity is None
    assert demographics.marital_status is None
    assert demographics.deceased is False


def test_contact_normalization():
    patient_data = {
        "address": [{"line": ["123 Main St"], "city": "Boston", "state": "MA", "postalCode": "02118", "country": "US"}],
        "telecom": [{"system": "phone", "value": "555-0000", "use": "home"}],
    }
    contact = normalize_contact(patient_data)

    assert contact.address == "123 Main St"
    assert contact.city == "Boston"
    assert contact.state == "MA"
    assert contact.postal_code == "02118"
    assert contact.country == "US"
    assert len(contact.telecom) == 1
    assert contact.telecom[0].value == "555-0000"


def test_contact_normalization_missing_fields():
    contact = normalize_contact({})

    assert contact.address is None
    assert contact.city is None
    assert contact.telecom == []


# --- condition ---------------------------------------------------------------


def test_condition_normalization():
    doc = _doc(
        "Condition",
        "cond-1",
        "p1",
        {
            "code": {"coding": [{"system": "http://snomed.info/sct", "code": "38341003", "display": "Hypertension"}], "text": "Hypertension"},
            "clinicalStatus": {"coding": [{"code": "active"}]},
            "verificationStatus": {"coding": [{"code": "confirmed"}]},
            "onsetDateTime": "2015-06-01",
            "recordedDate": "2015-06-02",
        },
    )
    condition = normalize_condition(doc)

    assert condition.resource_id == "cond-1"
    assert condition.code == "38341003"
    assert condition.display == "Hypertension"
    assert condition.system == "http://snomed.info/sct"
    # clinicalStatus/verificationStatus have no display/text in Synthea — the
    # bare code ("active"/"confirmed") is the correct, already human-readable fallback.
    assert condition.clinical_status == "active"
    assert condition.verification_status == "confirmed"
    assert condition.onset_date == "2015-06-01"
    assert condition.recorded_date == "2015-06-02"


def test_condition_normalization_missing_optional_fields():
    doc = _doc("Condition", "cond-2", "p1", {"code": {"text": "Some condition"}})
    condition = normalize_condition(doc)

    assert condition.resource_id == "cond-2"
    assert condition.display == "Some condition"
    assert condition.code is None
    assert condition.system is None
    assert condition.clinical_status is None
    assert condition.onset_date is None


# --- observation ---------------------------------------------------------------


def test_observation_normalization_quantity_value():
    doc = _doc(
        "Observation",
        "obs-1",
        "p1",
        {
            "status": "final",
            "code": {"coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic Blood Pressure"}], "text": "Systolic Blood Pressure"},
            "effectiveDateTime": "2023-01-10",
            "valueQuantity": {"value": 125, "unit": "mm[Hg]"},
            "referenceRange": [{"low": {"value": 90, "unit": "mm[Hg]"}, "high": {"value": 120, "unit": "mm[Hg]"}}],
            "interpretation": [{"coding": [{"code": "H"}]}],
        },
    )
    obs = normalize_observation(doc)

    assert obs.resource_id == "obs-1"
    assert obs.code == "8480-6"
    assert obs.name == "Systolic Blood Pressure"
    assert obs.system == "http://loinc.org"
    assert obs.value == 125
    assert obs.value_type == "Quantity"
    assert obs.unit == "mm[Hg]"
    assert obs.reference_range == "90-120 mm[Hg]"
    assert obs.interpretation == "H"
    assert obs.effective_date == "2023-01-10"
    assert obs.status == "final"


def test_observation_normalization_codeable_concept_value():
    doc = _doc(
        "Observation",
        "obs-2",
        "p1",
        {"status": "final", "code": {"text": "Appearance"}, "valueCodeableConcept": {"text": "Clear"}},
    )
    obs = normalize_observation(doc)

    assert obs.value == "Clear"
    assert obs.value_type == "CodeableConcept"
    assert obs.unit is None


def test_observation_normalization_missing_value():
    doc = _doc("Observation", "obs-3", "p1", {"status": "preliminary", "code": {"text": "Something"}})
    obs = normalize_observation(doc)

    assert obs.value is None
    assert obs.value_type is None
    assert obs.reference_range is None
    assert obs.interpretation is None


# --- medication ---------------------------------------------------------------


def test_medication_normalization():
    doc = _doc(
        "MedicationRequest",
        "med-1",
        "p1",
        {
            "status": "active",
            "intent": "order",
            "medicationCodeableConcept": {
                "coding": [{"system": "http://www.nlm.nih.gov/research/umls/rxnorm", "code": "860975", "display": "Metformin"}],
                "text": "Metformin 500 MG",
            },
            "authoredOn": "2022-03-01",
            "dosageInstruction": [
                {
                    "text": "Take 1 tablet twice daily",
                    "route": {"coding": [{"code": "PO", "display": "Oral"}]},
                    "timing": {"repeat": {"frequency": 2, "period": 1, "periodUnit": "d"}},
                }
            ],
            "dispenseRequest": {"validityPeriod": {"end": "2023-03-01"}},
        },
    )
    med = normalize_medication(doc)

    assert med.resource_id == "med-1"
    assert med.medication_name == "Metformin 500 MG"
    assert med.code == "860975"
    assert med.system == "http://www.nlm.nih.gov/research/umls/rxnorm"
    assert med.status == "active"
    assert med.intent == "order"
    assert med.dosage_text == "Take 1 tablet twice daily"
    assert med.route == "Oral"
    assert med.frequency == "2 time(s) per 1d"
    assert med.start_date == "2022-03-01"
    assert med.end_date == "2023-03-01"


def test_medication_normalization_missing_optional_fields():
    doc = _doc("MedicationRequest", "med-2", "p1", {"status": "stopped", "intent": "order"})
    med = normalize_medication(doc)

    assert med.medication_name is None
    assert med.dosage_text is None
    assert med.route is None
    assert med.frequency is None
    assert med.end_date is None


# --- procedure ---------------------------------------------------------------


def test_procedure_normalization():
    doc = _doc(
        "Procedure",
        "proc-1",
        "p1",
        {
            "status": "completed",
            "code": {"coding": [{"system": "http://snomed.info/sct", "code": "6025007", "display": "Appendectomy"}], "text": "Appendectomy"},
            "performedPeriod": {"start": "2019-08-15", "end": "2019-08-15"},
            "reasonCode": [{"text": "Appendicitis"}],
        },
    )
    proc = normalize_procedure(doc)

    assert proc.resource_id == "proc-1"
    assert proc.code == "6025007"
    assert proc.name == "Appendectomy"
    assert proc.system == "http://snomed.info/sct"
    assert proc.status == "completed"
    assert proc.performed_date == "2019-08-15"
    assert proc.reason == "Appendicitis"


def test_procedure_normalization_missing_optional_fields():
    doc = _doc("Procedure", "proc-2", "p1", {"status": "completed", "code": {"text": "Minor procedure"}})
    proc = normalize_procedure(doc)

    assert proc.performed_date is None
    assert proc.reason is None


# --- encounter ---------------------------------------------------------------


def test_encounter_normalization():
    doc = _doc(
        "Encounter",
        "enc-1",
        "p1",
        {
            "status": "finished",
            "type": [{"coding": [{"code": "185349003", "display": "Ambulatory"}], "text": "Ambulatory"}],
            "period": {"start": "2023-01-10T08:00:00-05:00", "end": "2023-01-10T09:00:00-05:00"},
            "reasonCode": [{"text": "Routine checkup"}],
            "location": [{"location": {"display": "Boston Clinic"}}],
        },
    )
    enc = normalize_encounter(doc)

    assert enc.resource_id == "enc-1"
    assert enc.encounter_type == "Ambulatory"
    assert enc.status == "finished"
    assert enc.start_date == "2023-01-10T08:00:00-05:00"
    assert enc.end_date == "2023-01-10T09:00:00-05:00"
    assert enc.reason == "Routine checkup"
    assert enc.location == "Boston Clinic"


def test_encounter_normalization_missing_optional_fields():
    doc = _doc("Encounter", "enc-2", "p1", {"status": "finished"})
    enc = normalize_encounter(doc)

    assert enc.encounter_type is None
    assert enc.start_date is None
    assert enc.reason is None
    assert enc.location is None


# --- diagnostic report ---------------------------------------------------------------


def test_diagnostic_report_normalization():
    doc = _doc(
        "DiagnosticReport",
        "dr-1",
        "p1",
        {
            "status": "final",
            "code": {"text": "CBC Panel"},
            "category": [{"coding": [{"code": "LAB", "display": "Laboratory"}]}],
            "effectiveDateTime": "2023-01-10",
            "result": [{"reference": "Observation/obs-1"}, {"reference": "urn:uuid:obs-2"}],
        },
    )
    report = normalize_diagnostic_report(doc)

    assert report.resource_id == "dr-1"
    assert report.report_name == "CBC Panel"
    assert report.status == "final"
    assert report.category == "Laboratory"
    assert report.effective_date == "2023-01-10"
    assert report.result_references == ["obs-1", "obs-2"]


def test_diagnostic_report_normalization_missing_optional_fields():
    doc = _doc("DiagnosticReport", "dr-2", "p1", {"status": "final", "code": {"text": "Note"}})
    report = normalize_diagnostic_report(doc)

    assert report.category is None
    assert report.result_references == []


# --- allergy ---------------------------------------------------------------


def test_allergy_normalization():
    doc = _doc(
        "AllergyIntolerance",
        "allergy-1",
        "p1",
        {
            "clinicalStatus": {"coding": [{"code": "active"}]},
            "verificationStatus": {"coding": [{"code": "confirmed"}]},
            "code": {"coding": [{"system": "http://snomed.info/sct", "code": "91935009", "display": "Peanut"}], "text": "Peanut allergy"},
            "reaction": [
                {"manifestation": [{"coding": [{"code": "39579001", "display": "Anaphylaxis"}]}], "severity": "severe"},
                {"manifestation": [{"text": "Rash"}], "severity": "mild"},
            ],
        },
    )
    allergy = normalize_allergy(doc)

    assert allergy.resource_id == "allergy-1"
    assert allergy.substance == "Peanut allergy"
    assert allergy.code == "91935009"
    assert allergy.system == "http://snomed.info/sct"
    assert allergy.clinical_status == "active"
    assert allergy.verification_status == "confirmed"
    assert len(allergy.reaction) == 2
    assert allergy.reaction[0].manifestation == ["Anaphylaxis"]
    assert allergy.reaction[0].severity == "severe"
    assert allergy.severity == "severe"  # convenience: first reaction's severity
    assert allergy.manifestation == ["Anaphylaxis", "Rash"]  # convenience: flattened across all reactions


def test_allergy_normalization_missing_optional_fields():
    doc = _doc("AllergyIntolerance", "allergy-2", "p1", {"code": {"text": "Unknown allergen"}})
    allergy = normalize_allergy(doc)

    assert allergy.substance == "Unknown allergen"
    assert allergy.clinical_status is None
    assert allergy.reaction == []
    assert allergy.severity is None
    assert allergy.manifestation == []


# --- build_patient_profile / cross-patient isolation -------------------------


def test_build_patient_profile_associates_only_matching_resources():
    patient_doc = {"patient_id": "p1", "data": {"id": "p1", "name": [{"family": "Doe", "given": ["Jane"]}]}}
    resource_docs = [
        _doc("Condition", "cond-1", "p1", {"code": {"text": "Hypertension"}}),
        _doc("Observation", "obs-1", "p1", {"code": {"text": "Heart rate"}, "status": "final"}),
    ]

    profile = build_patient_profile(patient_doc, resource_docs)

    assert profile.patient_id == "p1"
    assert len(profile.conditions) == 1
    assert len(profile.observations) == 1
    assert profile.demographics.first_name == "Jane"


def test_build_patient_profile_ignores_resources_from_another_patient():
    # Defensive isolation: even if a foreign-patient resource is (incorrectly)
    # present in the input list, it must never leak into the profile.
    patient_doc = {"patient_id": "p1", "data": {"id": "p1"}}
    resource_docs = [
        _doc("Condition", "cond-1", "p1", {"code": {"text": "Hypertension"}}),
        _doc("Condition", "cond-2", "OTHER-PATIENT", {"code": {"text": "Diabetes"}}),
    ]

    profile = build_patient_profile(patient_doc, resource_docs)

    assert len(profile.conditions) == 1
    assert profile.conditions[0].resource_id == "cond-1"


def test_build_patient_profile_ignores_unsupported_resource_types():
    patient_doc = {"patient_id": "p1", "data": {"id": "p1"}}
    resource_docs = [_doc("Immunization", "imm-1", "p1", {"status": "completed"})]

    profile = build_patient_profile(patient_doc, resource_docs)

    assert profile.conditions == []
    assert profile.observations == []
    assert profile.medications == []
