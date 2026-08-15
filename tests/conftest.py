from pathlib import Path

import mongomock
import pytest
from fastapi.testclient import TestClient

from app.db.mongodb import ensure_indexes, get_database
from app.main import app


@pytest.fixture
def mongo_db():
    client = mongomock.MongoClient()
    db = client["test_clinical_trial_matching"]
    ensure_indexes(db)
    return db


@pytest.fixture
def api_client(mongo_db):
    app.dependency_overrides[get_database] = lambda: mongo_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"
