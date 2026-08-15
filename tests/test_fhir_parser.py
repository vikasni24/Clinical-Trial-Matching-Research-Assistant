from app.services.fhir_parser import extract_resources


def test_standalone_resource_parsing():
    raw = {"resourceType": "Patient", "id": "patient-001", "gender": "female"}
    outcomes = extract_resources(raw, source_file="patient.json")

    assert len(outcomes) == 1
    assert outcomes[0].status == "parsed"
    assert outcomes[0].resource.resource_type == "Patient"
    assert outcomes[0].resource.resource_id == "patient-001"
    assert outcomes[0].resource.patient_id == "patient-001"


def test_bundle_parsing_extracts_multiple_resources():
    raw = {
        "resourceType": "Bundle",
        "id": "bundle-1",
        "entry": [
            {"resource": {"resourceType": "Patient", "id": "p1"}},
            {"resource": {"resourceType": "Observation", "id": "o1", "subject": {"reference": "Patient/p1"}}},
        ],
    }
    outcomes = extract_resources(raw, source_file="bundle.json")
    parsed = [o for o in outcomes if o.status == "parsed"]

    assert len(parsed) == 2
    assert parsed[0].resource.source_bundle_id == "bundle-1"
    assert parsed[1].resource.patient_id == "p1"


def test_patient_association_via_subject_reference():
    raw = {"resourceType": "Condition", "id": "c1", "subject": {"reference": "Patient/patient-xyz"}}
    outcomes = extract_resources(raw, source_file="c.json")

    assert outcomes[0].resource.patient_id == "patient-xyz"


def test_patient_association_via_urn_uuid_reference():
    # Synthea commonly references patients as "urn:uuid:<id>" rather than "Patient/<id>".
    raw = {"resourceType": "Condition", "id": "c1", "subject": {"reference": "urn:uuid:patient-xyz"}}
    outcomes = extract_resources(raw, source_file="c.json")

    assert outcomes[0].resource.patient_id == "patient-xyz"


def test_missing_resource_id_is_flagged_failed():
    raw = {"resourceType": "Observation", "status": "final"}
    outcomes = extract_resources(raw, source_file="o.json")

    assert outcomes[0].status == "failed"
    assert "missing" in outcomes[0].message.lower()


def test_unsupported_resource_type_is_skipped():
    raw = {"resourceType": "Coverage", "id": "cov-1"}
    outcomes = extract_resources(raw, source_file="cov.json")

    assert outcomes[0].status == "skipped_unsupported"


def test_bundle_entry_missing_resource_is_a_failure():
    raw = {"resourceType": "Bundle", "id": "b1", "entry": [{"fullUrl": "urn:uuid:1"}]}
    outcomes = extract_resources(raw, source_file="b.json")

    assert outcomes[0].status == "failed"


def test_non_object_root_is_a_failure():
    outcomes = extract_resources(["not", "an", "object"], source_file="weird.json")

    assert outcomes[0].status == "failed"
