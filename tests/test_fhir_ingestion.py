import json
import shutil
from pathlib import Path

from app.services.fhir_ingestion import FHIRIngestionService


def _copy_fixture(fixtures_dir: Path, name: str, dest_dir: Path) -> None:
    shutil.copy(fixtures_dir / name, dest_dir / name)


def test_ingest_standalone_and_bundle(tmp_path, mongo_db, fixtures_dir):
    _copy_fixture(fixtures_dir, "patient_standalone.json", tmp_path)
    _copy_fixture(fixtures_dir, "bundle_sample.json", tmp_path)

    stats = FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()

    assert stats.files_discovered == 2
    assert stats.files_processed == 2
    assert stats.files_failed == 0
    assert stats.resources_inserted == 4  # patient-001, patient-002, obs-100, cond-200
    assert stats.resources_skipped == 1  # Coverage is unsupported
    assert stats.resources_failed == 0

    assert mongo_db["patients"].count_documents({}) == 2
    assert mongo_db["fhir_resources"].count_documents({}) == 4


def test_invalid_json_does_not_crash_ingestion(tmp_path, mongo_db, fixtures_dir):
    _copy_fixture(fixtures_dir, "invalid.json", tmp_path)
    _copy_fixture(fixtures_dir, "patient_standalone.json", tmp_path)

    stats = FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()

    assert stats.files_discovered == 2
    assert stats.files_failed == 1
    assert stats.files_processed == 1
    assert stats.resources_inserted == 1
    assert len(stats.errors) >= 1


def test_missing_resource_id_is_counted_failed(tmp_path, mongo_db, fixtures_dir):
    _copy_fixture(fixtures_dir, "missing_id.json", tmp_path)

    stats = FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()

    assert stats.resources_failed == 1
    assert stats.resources_inserted == 0


def test_original_fhir_json_is_preserved(tmp_path, mongo_db, fixtures_dir):
    _copy_fixture(fixtures_dir, "patient_standalone.json", tmp_path)
    original = json.loads((fixtures_dir / "patient_standalone.json").read_text())

    FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()

    stored = mongo_db["fhir_resources"].find_one({"resource_type": "Patient", "resource_id": "patient-001"})
    assert stored["data"] == original
    assert stored["source_file"] == "patient_standalone.json"
    assert "ingested_at" in stored


def test_rerunning_ingestion_upserts_not_duplicates(tmp_path, mongo_db, fixtures_dir):
    _copy_fixture(fixtures_dir, "patient_standalone.json", tmp_path)
    _copy_fixture(fixtures_dir, "bundle_sample.json", tmp_path)

    service = FHIRIngestionService(mongo_db, fhir_dir=tmp_path)
    first_stats = service.run()
    second_stats = service.run()

    assert first_stats.resources_inserted == 4
    assert second_stats.resources_inserted == 0
    assert second_stats.resources_updated == 4
    assert mongo_db["fhir_resources"].count_documents({}) == 4
    assert mongo_db["patients"].count_documents({}) == 2


def test_missing_directory_returns_zero_stats_without_crashing(tmp_path, mongo_db):
    stats = FHIRIngestionService(mongo_db, fhir_dir=tmp_path / "does-not-exist").run()

    assert stats.files_discovered == 0
    assert stats.resources_processed == 0
