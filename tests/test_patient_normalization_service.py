import shutil

from app.repositories import patient_profile_repository
from app.services.fhir_ingestion import FHIRIngestionService
from app.services.patient_normalization import PatientNormalizationService


def _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    return FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()


def test_normalize_patient_creates_profile_from_real_ingested_patient(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    profile = PatientNormalizationService(mongo_db).normalize_patient("profile-patient-1")

    assert profile is not None
    assert profile.patient_id == "profile-patient-1"
    assert profile.demographics.full_name == "Jane Q Doe"
    assert profile.contact.city == "Boston"
    assert len(profile.conditions) == 1
    assert len(profile.observations) == 1
    assert len(profile.medications) == 1
    assert len(profile.procedures) == 1
    assert len(profile.encounters) == 1
    assert len(profile.diagnostic_reports) == 1
    assert len(profile.allergies) == 1

    stored = patient_profile_repository.get_patient_profile(mongo_db, "profile-patient-1")
    assert stored is not None
    assert stored["demographics"]["full_name"] == "Jane Q Doe"


def test_normalize_patient_unknown_patient_returns_none(mongo_db):
    profile = PatientNormalizationService(mongo_db).normalize_patient("does-not-exist")

    assert profile is None
    assert patient_profile_repository.get_patient_profile(mongo_db, "does-not-exist") is None


def test_normalize_patient_isolates_cross_patient_data(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    profile_1 = PatientNormalizationService(mongo_db).normalize_patient("profile-patient-1")
    profile_2 = PatientNormalizationService(mongo_db).normalize_patient("profile-patient-2")

    assert [c.resource_id for c in profile_1.conditions] == ["cond-1"]
    assert [c.resource_id for c in profile_2.conditions] == ["cond-2"]
    # profile-patient-2's condition (diabetes) must never appear on patient 1's profile.
    assert "cond-2" not in [c.resource_id for c in profile_1.conditions]
    assert "cond-1" not in [c.resource_id for c in profile_2.conditions]


def test_normalize_patient_upsert_is_idempotent(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)
    service = PatientNormalizationService(mongo_db)

    service.normalize_patient("profile-patient-1")
    service.normalize_patient("profile-patient-1")

    assert patient_profile_repository.count_patient_profiles(mongo_db) == 1


def test_normalize_all_batch_normalizes_every_patient(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    stats = PatientNormalizationService(mongo_db).normalize_all()

    assert stats.patients_processed == 2
    assert stats.profiles_inserted == 2
    assert stats.profiles_updated == 0
    assert stats.profiles_failed == 0
    assert patient_profile_repository.count_patient_profiles(mongo_db) == 2


def test_normalize_all_rerun_upserts_not_duplicates(mongo_db, tmp_path, fixtures_dir):
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)
    service = PatientNormalizationService(mongo_db)

    first_stats = service.normalize_all()
    second_stats = service.normalize_all()

    assert first_stats.profiles_inserted == 2
    assert second_stats.profiles_inserted == 0
    assert second_stats.profiles_updated == 2
    assert patient_profile_repository.count_patient_profiles(mongo_db) == 2


def test_normalize_all_ignores_unsupported_resources(mongo_db, tmp_path, fixtures_dir):
    # profile_bundle.json includes an Immunization (unsupported, skipped by Phase 1
    # ingestion already) — normalization must not fail or fabricate data for it.
    _ingest_profile_bundle(mongo_db, tmp_path, fixtures_dir)

    profile = PatientNormalizationService(mongo_db).normalize_patient("profile-patient-1")

    assert profile is not None
