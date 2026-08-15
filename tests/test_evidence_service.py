"""Tests for EvidenceService: a thin orchestration layer over
evidence_repository.

Delegation/separation is proven with mocks (patching the repository
functions the service calls); functional/behavioral correctness is proven
with mongomock + real ingestion, mirroring the project's existing test
conventions (see tests/test_evidence_repository.py)."""

import inspect
import shutil
from unittest.mock import patch

import pytest

from app.models.evidence import Evidence
from app.services.evidence_service import EvidenceService
from app.services.fhir_ingestion import FHIRIngestionService


def _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    return FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()


# --- delegation to the repository (mocked) -----------------------------------


def test_get_patient_evidence_delegates_to_repository():
    fake_db = object()
    sentinel = iter([])
    with patch(
        "app.services.evidence_service.evidence_repository.get_patient_evidence", return_value=sentinel
    ) as mocked:
        result = EvidenceService(fake_db).get_patient_evidence("p1", resource_type="Condition")

    mocked.assert_called_once_with(fake_db, "p1", resource_type="Condition")
    assert result is sentinel


def test_get_patient_observation_evidence_delegates_to_repository():
    fake_db = object()
    sentinel = iter([])
    with patch(
        "app.services.evidence_service.evidence_repository.get_patient_observation_evidence", return_value=sentinel
    ) as mocked:
        result = EvidenceService(fake_db).get_patient_observation_evidence("p1")

    mocked.assert_called_once_with(fake_db, "p1")
    assert result is sentinel


def test_get_patient_evidence_by_code_delegates_to_repository():
    fake_db = object()
    sentinel = iter([])
    with patch(
        "app.services.evidence_service.evidence_repository.get_patient_evidence_by_code", return_value=sentinel
    ) as mocked:
        result = EvidenceService(fake_db).get_patient_evidence_by_code("p1", "38341003", resource_type="Condition")

    mocked.assert_called_once_with(fake_db, "p1", "38341003", resource_type="Condition")
    assert result is sentinel


def test_get_patient_resource_evidence_delegates_to_repository():
    fake_db = object()
    sentinel_evidence = Evidence(patient_id="p1", resource_type="Observation", resource_id="obs-1")
    with patch(
        "app.services.evidence_service.evidence_repository.get_patient_resource_evidence",
        return_value=sentinel_evidence,
    ) as mocked:
        result = EvidenceService(fake_db).get_patient_resource_evidence("p1", "Observation", "obs-1")

    mocked.assert_called_once_with(fake_db, "p1", "Observation", "obs-1")
    assert result is sentinel_evidence


def test_service_never_queries_mongodb_directly():
    # Structural proof the service can't be building its own Mongo queries:
    # it doesn't reference the collection name, `.find(`, or `db[`.
    import app.services.evidence_service as module

    source = inspect.getsource(module)
    assert "FHIR_RESOURCES_COLLECTION" not in source
    assert ".find(" not in source
    assert "db[" not in source


def test_repository_errors_are_not_silently_swallowed():
    class Boom(Exception):
        pass

    with patch(
        "app.services.evidence_service.evidence_repository.get_patient_evidence",
        side_effect=Boom("simulated repository failure"),
    ):
        with pytest.raises(Boom):
            EvidenceService(object()).get_patient_evidence("p1")


# --- functional / integration behavior (mongomock + real ingestion) ----------


def test_patient_evidence_retrieval_works(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)
    service = EvidenceService(mongo_db)

    items = list(service.get_patient_evidence("profile-patient-1"))

    assert len(items) > 0
    assert all(e.patient_id == "profile-patient-1" for e in items)


def test_resource_type_filtering_works(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)
    service = EvidenceService(mongo_db)

    items = list(service.get_patient_evidence("profile-patient-1", resource_type="Condition"))

    assert len(items) == 1
    assert items[0].resource_type == "Condition"
    assert items[0].resource_id == "cond-1"


def test_observation_retrieval_works(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)
    service = EvidenceService(mongo_db)

    items = list(service.get_patient_observation_evidence("profile-patient-1"))

    assert len(items) == 1
    assert items[0].resource_type == "Observation"
    assert items[0].value == 125
    assert items[0].unit == "mm[Hg]"


def test_code_based_retrieval_works(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)
    service = EvidenceService(mongo_db)

    items = list(service.get_patient_evidence_by_code("profile-patient-1", "38341003"))

    assert len(items) == 1
    assert items[0].display == "Hypertension"


def test_specific_resource_retrieval_works(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)
    service = EvidenceService(mongo_db)

    evidence = service.get_patient_resource_evidence("profile-patient-1", "Observation", "obs-1")

    assert evidence is not None
    assert evidence.resource_id == "obs-1"


def test_patient_isolation(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)
    service = EvidenceService(mongo_db)

    patient_1_items = list(service.get_patient_evidence("profile-patient-1"))
    patient_2_items = list(service.get_patient_evidence("profile-patient-2"))

    assert all(e.patient_id == "profile-patient-1" for e in patient_1_items)
    assert all(e.patient_id == "profile-patient-2" for e in patient_2_items)
    ids_1 = {e.resource_id for e in patient_1_items}
    ids_2 = {e.resource_id for e in patient_2_items}
    assert ids_1.isdisjoint(ids_2)

    # cond-2 (Diabetes) belongs to profile-patient-2 — requesting it under
    # profile-patient-1's id must never return it.
    leaked = service.get_patient_resource_evidence("profile-patient-1", "Condition", "cond-2")
    assert leaked is None


def test_unknown_patient_returns_empty_iterable(mongo_db):
    service = EvidenceService(mongo_db)

    items = list(service.get_patient_evidence("does-not-exist"))

    assert items == []


def test_no_matching_evidence_returns_empty_iterable(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)
    service = EvidenceService(mongo_db)

    items = list(service.get_patient_evidence_by_code("profile-patient-1", "does-not-exist-code"))

    assert items == []


def test_missing_specific_resource_returns_none(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)
    service = EvidenceService(mongo_db)

    evidence = service.get_patient_resource_evidence("profile-patient-1", "Observation", "does-not-exist")

    assert evidence is None


def test_evidence_traceability_is_preserved(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)
    service = EvidenceService(mongo_db)

    evidence = service.get_patient_resource_evidence("profile-patient-1", "Observation", "obs-1")

    assert evidence.patient_id == "profile-patient-1"
    assert evidence.resource_type == "Observation"
    assert evidence.resource_id == "obs-1"
    assert evidence.source_collection == "fhir_resources"
    assert evidence.source_reference == "Observation/obs-1"


def test_missing_clinical_values_remain_none(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)
    service = EvidenceService(mongo_db)

    # DiagnosticReport dr-1 in the fixture has code.text but no coding[].
    evidence = service.get_patient_resource_evidence("profile-patient-1", "DiagnosticReport", "dr-1")

    assert evidence.code is None
    assert evidence.coding_system is None


def test_service_does_not_invent_evidence(mongo_db):
    # A patient that was never ingested has zero real resources — the
    # service must never fabricate placeholder evidence for them.
    service = EvidenceService(mongo_db)

    items = list(service.get_patient_evidence("phantom-patient"))
    single = service.get_patient_resource_evidence("phantom-patient", "Observation", "phantom-obs")

    assert items == []
    assert single is None


def test_repository_generator_laziness_is_preserved(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)
    service = EvidenceService(mongo_db)

    result = service.get_patient_evidence("profile-patient-1")

    assert inspect.isgenerator(result)

    result_by_code = service.get_patient_evidence_by_code("profile-patient-1", "38341003")
    assert inspect.isgenerator(result_by_code)


def test_deterministic_repeated_calls_produce_identical_results(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)
    service = EvidenceService(mongo_db)

    first = [e.model_dump() for e in service.get_patient_evidence("profile-patient-1")]
    second = [e.model_dump() for e in service.get_patient_evidence("profile-patient-1")]

    assert first == second
