import shutil

from app.services.trial_ingestion import TrialIngestionService


def test_get_trials_empty(api_client):
    resp = api_client.get("/api/trials")

    assert resp.status_code == 200
    body = resp.json()
    assert body["items"] == []
    assert "disclaimer" in body
    assert "synthetic" in body["disclaimer"].lower()


def test_ingest_and_list_trials(api_client, mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "trials_valid.json", tmp_path / "trials_valid.json")
    TrialIngestionService(mongo_db, trials_path=tmp_path / "trials_valid.json").run()

    resp = api_client.get("/api/trials")

    assert resp.status_code == 200
    body = resp.json()
    assert body["pagination"]["total"] == 2


def test_get_trial_by_id(api_client, mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "trials_valid.json", tmp_path / "trials_valid.json")
    TrialIngestionService(mongo_db, trials_path=tmp_path / "trials_valid.json").run()

    resp = api_client.get("/api/trials/TEST-TRIAL-A")

    assert resp.status_code == 200
    body = resp.json()
    assert body["trial_id"] == "TEST-TRIAL-A"
    assert body["source"] == "synthetic_dev_fixture"


def test_get_trial_not_found(api_client):
    resp = api_client.get("/api/trials/does-not-exist")

    assert resp.status_code == 404


def test_trial_pagination(api_client, mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "trials_valid.json", tmp_path / "trials_valid.json")
    TrialIngestionService(mongo_db, trials_path=tmp_path / "trials_valid.json").run()

    resp = api_client.get("/api/trials?page=1&page_size=1")

    body = resp.json()
    assert len(body["items"]) == 1
    assert body["pagination"]["total_pages"] == 2


def test_trial_status_filter(api_client, mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "trials_valid.json", tmp_path / "trials_valid.json")
    TrialIngestionService(mongo_db, trials_path=tmp_path / "trials_valid.json").run()

    resp = api_client.get("/api/trials?status=completed")

    body = resp.json()
    assert body["pagination"]["total"] == 1
    assert body["items"][0]["trial_id"] == "TEST-TRIAL-B"


def test_post_ingest_trials_uses_real_dev_fixture(api_client, mongo_db):
    # Exercises the actual shipped data/trials/dev_trials.json end-to-end.
    resp = api_client.post("/api/trials/ingest")

    assert resp.status_code == 200
    body = resp.json()
    assert body["trials_discovered"] == 17
    assert body["trials_failed"] == 0
    assert "synthetic" in body["disclaimer"].lower()


def test_post_ingest_trials_is_idempotent(api_client, mongo_db):
    first = api_client.post("/api/trials/ingest").json()
    second = api_client.post("/api/trials/ingest").json()

    assert first["trials_inserted"] == 17
    assert second["trials_inserted"] == 0
    assert second["trials_updated"] == 17
