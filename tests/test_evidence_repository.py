import inspect
import shutil

from app.repositories import evidence_repository
from app.services.fhir_ingestion import FHIRIngestionService


def _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    return FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()


# --- patient-scoped retrieval / resource-type filtering ---------------------


def test_get_patient_evidence_is_scoped_to_patient(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    items = list(evidence_repository.get_patient_evidence(mongo_db, "profile-patient-1"))

    assert len(items) > 0
    assert all(e.patient_id == "profile-patient-1" for e in items)


def test_get_patient_evidence_resource_type_filter(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    items = list(evidence_repository.get_patient_evidence(mongo_db, "profile-patient-1", resource_type="Condition"))

    assert len(items) == 1
    assert items[0].resource_type == "Condition"
    assert items[0].resource_id == "cond-1"


def test_get_patient_observation_evidence(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    items = list(evidence_repository.get_patient_observation_evidence(mongo_db, "profile-patient-1"))

    assert len(items) == 1
    assert items[0].resource_type == "Observation"
    assert items[0].value == 125
    assert items[0].unit == "mm[Hg]"


# --- code-based retrieval ----------------------------------------------------


def test_get_patient_evidence_by_code(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    items = list(evidence_repository.get_patient_evidence_by_code(mongo_db, "profile-patient-1", code="38341003"))

    assert len(items) == 1
    assert items[0].resource_type == "Condition"
    assert items[0].display == "Hypertension"


def test_get_patient_evidence_by_code_with_resource_type(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    items = list(
        evidence_repository.get_patient_evidence_by_code(
            mongo_db, "profile-patient-1", code="860975", resource_type="MedicationRequest"
        )
    )

    assert len(items) == 1
    assert items[0].resource_type == "MedicationRequest"
    assert items[0].display == "Metformin 500 MG"


def test_get_patient_evidence_by_code_missing_returns_empty(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    items = list(evidence_repository.get_patient_evidence_by_code(mongo_db, "profile-patient-1", code="does-not-exist"))

    assert items == []


# --- specific resource retrieval ---------------------------------------------


def test_get_patient_resource_evidence(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    evidence = evidence_repository.get_patient_resource_evidence(mongo_db, "profile-patient-1", "Observation", "obs-1")

    assert evidence is not None
    assert evidence.resource_id == "obs-1"
    assert evidence.source_collection == "fhir_resources"
    assert evidence.source_reference == "Observation/obs-1"


def test_get_patient_resource_evidence_missing_resource_returns_none(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    evidence = evidence_repository.get_patient_resource_evidence(
        mongo_db, "profile-patient-1", "Observation", "does-not-exist"
    )

    assert evidence is None


def test_get_patient_resource_evidence_wrong_patient_returns_none(mongo_db, tmp_path, fixtures_dir):
    # obs-1 belongs to profile-patient-1 — requesting it under a different
    # patient_id must never return it.
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    evidence = evidence_repository.get_patient_resource_evidence(mongo_db, "profile-patient-2", "Observation", "obs-1")

    assert evidence is None


# --- patient isolation --------------------------------------------------------


def test_patient_isolation_across_patients(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    patient_1_items = list(evidence_repository.get_patient_evidence(mongo_db, "profile-patient-1"))
    patient_2_items = list(evidence_repository.get_patient_evidence(mongo_db, "profile-patient-2"))

    assert all(e.patient_id == "profile-patient-1" for e in patient_1_items)
    assert all(e.patient_id == "profile-patient-2" for e in patient_2_items)

    ids_1 = {e.resource_id for e in patient_1_items}
    ids_2 = {e.resource_id for e in patient_2_items}
    assert ids_1.isdisjoint(ids_2)
    assert "cond-2" not in ids_1  # patient-2's Diabetes condition must never appear for patient-1
    assert "cond-1" not in ids_2  # patient-1's Hypertension condition must never appear for patient-2


def test_missing_patient_returns_empty_result(mongo_db):
    items = list(evidence_repository.get_patient_evidence(mongo_db, "does-not-exist"))

    assert items == []


# --- observation value extraction --------------------------------------------


def test_observation_value_extraction(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    evidence = evidence_repository.get_patient_resource_evidence(mongo_db, "profile-patient-1", "Observation", "obs-1")

    assert evidence.code == "8480-6"
    assert evidence.coding_system == "http://loinc.org"
    assert evidence.value == 125
    assert evidence.unit == "mm[Hg]"
    assert evidence.effective_date == "2023-01-10"
    assert evidence.status == "final"


def test_observation_without_value_preserves_none(mongo_db):
    document = {
        "patient_id": "p1",
        "resource_type": "Observation",
        "resource_id": "obs-no-value",
        "data": {
            "resourceType": "Observation",
            "id": "obs-no-value",
            "status": "preliminary",
            "code": {"text": "Some test"},
        },
    }
    mongo_db["fhir_resources"].insert_one(document)

    evidence = evidence_repository.get_patient_resource_evidence(mongo_db, "p1", "Observation", "obs-no-value")

    assert evidence.value is None
    assert evidence.unit is None
    assert evidence.status == "preliminary"


# --- resource-type extraction -------------------------------------------------


def test_condition_extraction(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    evidence = evidence_repository.get_patient_resource_evidence(mongo_db, "profile-patient-1", "Condition", "cond-1")

    assert evidence.code == "38341003"
    assert evidence.coding_system == "http://snomed.info/sct"
    assert evidence.display == "Hypertension"
    assert evidence.status == "active"
    assert evidence.effective_date == "2015-06-01"


def test_medication_request_extraction(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    evidence = evidence_repository.get_patient_resource_evidence(
        mongo_db, "profile-patient-1", "MedicationRequest", "med-1"
    )

    assert evidence.code == "860975"
    assert evidence.coding_system == "http://www.nlm.nih.gov/research/umls/rxnorm"
    assert evidence.display == "Metformin 500 MG"
    assert evidence.status == "active"
    assert evidence.effective_date == "2022-03-01"


def test_allergy_intolerance_extraction(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    evidence = evidence_repository.get_patient_resource_evidence(
        mongo_db, "profile-patient-1", "AllergyIntolerance", "allergy-1"
    )

    assert evidence.code == "91935009"
    assert evidence.display == "Peanut allergy"
    assert evidence.status == "active"


def test_procedure_extraction(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    evidence = evidence_repository.get_patient_resource_evidence(mongo_db, "profile-patient-1", "Procedure", "proc-1")

    assert evidence.code == "6025007"
    assert evidence.display == "Appendectomy"
    assert evidence.status == "completed"
    assert evidence.effective_date == "2019-08-15"


def test_encounter_extraction(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    evidence = evidence_repository.get_patient_resource_evidence(mongo_db, "profile-patient-1", "Encounter", "enc-1")

    assert evidence.code == "185349003"
    assert evidence.display == "Ambulatory"
    assert evidence.status == "finished"
    assert evidence.effective_date == "2023-01-10T08:00:00-05:00"


def test_diagnostic_report_extraction(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    evidence = evidence_repository.get_patient_resource_evidence(
        mongo_db, "profile-patient-1", "DiagnosticReport", "dr-1"
    )

    assert evidence.display == "CBC Panel"  # code.text — this fixture's DiagnosticReport has no coding[]
    assert evidence.code is None  # honestly reflects the absence of a coding entry, not invented
    assert evidence.status == "final"
    assert evidence.effective_date == "2023-01-10"


def test_patient_demographics_extraction(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    evidence = evidence_repository.get_patient_resource_evidence(
        mongo_db, "profile-patient-1", "Patient", "profile-patient-1"
    )

    assert evidence.display == "Jane Q Doe"
    assert evidence.value == "female"
    assert evidence.effective_date == "1980-05-12"


# --- generator / scalability behavior -----------------------------------------


def test_get_patient_evidence_returns_a_generator(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    result = evidence_repository.get_patient_evidence(mongo_db, "profile-patient-1")

    assert inspect.isgenerator(result)
    assert not isinstance(result, list)


def test_get_patient_evidence_by_code_returns_a_generator(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    result = evidence_repository.get_patient_evidence_by_code(mongo_db, "profile-patient-1", code="38341003")

    assert inspect.isgenerator(result)


def test_queries_are_always_patient_scoped_never_unbounded(mongo_db, tmp_path, fixtures_dir, monkeypatch):
    # Behavioral proof that every query this repository issues includes a
    # patient_id filter and is never the empty/unbounded {} filter.
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    collection = mongo_db["fhir_resources"]
    original_find = collection.find
    captured_queries = []

    def spying_find(filter=None, *args, **kwargs):
        captured_queries.append(filter or {})
        return original_find(filter, *args, **kwargs)

    monkeypatch.setattr(collection, "find", spying_find)

    list(evidence_repository.get_patient_evidence(mongo_db, "profile-patient-1"))
    list(evidence_repository.get_patient_evidence(mongo_db, "profile-patient-1", resource_type="Observation"))
    list(evidence_repository.get_patient_evidence_by_code(mongo_db, "profile-patient-1", code="38341003"))

    assert len(captured_queries) == 3
    for query in captured_queries:
        assert query != {}
        assert query.get("patient_id") == "profile-patient-1"


def test_original_fhir_document_remains_unchanged(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    before = mongo_db["fhir_resources"].find_one({"resource_type": "Observation", "resource_id": "obs-1"})
    evidence_repository.get_patient_resource_evidence(mongo_db, "profile-patient-1", "Observation", "obs-1")
    list(evidence_repository.get_patient_evidence(mongo_db, "profile-patient-1"))
    after = mongo_db["fhir_resources"].find_one({"resource_type": "Observation", "resource_id": "obs-1"})

    assert before == after
