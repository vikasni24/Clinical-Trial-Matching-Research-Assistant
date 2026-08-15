import shutil

from app.repositories import trial_repository
from app.services.trial_ingestion import TrialIngestionService


def test_valid_trial_ingestion(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "trials_valid.json", tmp_path / "trials_valid.json")
    stats = TrialIngestionService(mongo_db, trials_path=tmp_path / "trials_valid.json").run()

    assert stats.trials_discovered == 2
    assert stats.trials_inserted == 2
    assert stats.trials_updated == 0
    assert stats.trials_failed == 0
    assert trial_repository.count_trials(mongo_db) == 2


def test_invalid_json_is_reported_and_does_not_crash(mongo_db, fixtures_dir, tmp_path):
    shutil.copy(fixtures_dir / "trials_invalid.json", tmp_path / "trials_invalid.json")
    stats = TrialIngestionService(mongo_db, trials_path=tmp_path / "trials_invalid.json").run()

    assert stats.trials_discovered == 0
    assert stats.trials_inserted == 0
    assert len(stats.errors) == 1
    assert "Invalid JSON" in stats.errors[0]
    assert trial_repository.count_trials(mongo_db) == 0


def test_missing_disclaimer_is_rejected(mongo_db, fixtures_dir, tmp_path):
    shutil.copy(fixtures_dir / "trials_no_disclaimer.json", tmp_path / "trials_no_disclaimer.json")
    stats = TrialIngestionService(mongo_db, trials_path=tmp_path / "trials_no_disclaimer.json").run()

    assert stats.trials_discovered == 0
    assert len(stats.errors) == 1
    assert "disclaimer" in stats.errors[0].lower()
    assert trial_repository.count_trials(mongo_db) == 0


def test_missing_trial_id_record_is_reported_as_failed(mongo_db, fixtures_dir, tmp_path):
    shutil.copy(fixtures_dir / "trials_missing_trial_id.json", tmp_path / "trials_missing_trial_id.json")
    stats = TrialIngestionService(mongo_db, trials_path=tmp_path / "trials_missing_trial_id.json").run()

    assert stats.trials_discovered == 1
    assert stats.trials_failed == 1
    assert stats.trials_inserted == 0


def test_rerunning_ingestion_upserts_not_duplicates(mongo_db, fixtures_dir, tmp_path):
    shutil.copy(fixtures_dir / "trials_valid.json", tmp_path / "trials_valid.json")
    service = TrialIngestionService(mongo_db, trials_path=tmp_path / "trials_valid.json")

    first = service.run()
    second = service.run()

    assert first.trials_inserted == 2
    assert second.trials_inserted == 0
    assert second.trials_updated == 2
    assert trial_repository.count_trials(mongo_db) == 2


def test_ingested_trial_source_is_always_synthetic_dev_fixture(mongo_db, fixtures_dir, tmp_path):
    shutil.copy(fixtures_dir / "trials_valid.json", tmp_path / "trials_valid.json")
    TrialIngestionService(mongo_db, trials_path=tmp_path / "trials_valid.json").run()

    trial = trial_repository.get_trial(mongo_db, "TEST-TRIAL-A")
    assert trial["source"] == "synthetic_dev_fixture"


def test_missing_fixture_file_is_reported_gracefully(mongo_db, tmp_path):
    stats = TrialIngestionService(mongo_db, trials_path=tmp_path / "does-not-exist.json").run()

    assert stats.trials_discovered == 0
    assert len(stats.errors) == 1
    assert trial_repository.count_trials(mongo_db) == 0


def test_trial_retrieval_after_ingestion(mongo_db, fixtures_dir, tmp_path):
    shutil.copy(fixtures_dir / "trials_valid.json", tmp_path / "trials_valid.json")
    TrialIngestionService(mongo_db, trials_path=tmp_path / "trials_valid.json").run()

    trial = trial_repository.get_trial(mongo_db, "TEST-TRIAL-B")
    assert trial is not None
    assert trial["status"] == "completed"
    assert trial["eligibility"]["maximum_age"] == 17
