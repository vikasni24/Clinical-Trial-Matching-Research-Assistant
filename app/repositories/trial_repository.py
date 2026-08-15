"""Data access layer for clinical_trials documents. Mirrors the conventions
in fhir_repository.py / patient_profile_repository.py."""

from typing import Any, Iterator, Optional

from pymongo.database import Database

from app.db.mongodb import CLINICAL_TRIALS_COLLECTION

_EXCLUDE_MONGO_ID = {"_id": 0}


def upsert_trial(db: Database, document: dict[str, Any]) -> str:
    """Insert or update a trial keyed on trial_id. Returns "inserted" or "updated"."""
    collection = db[CLINICAL_TRIALS_COLLECTION]
    filter_query = {"trial_id": document["trial_id"]}
    result = collection.update_one(filter_query, {"$set": document}, upsert=True)
    return "inserted" if result.upserted_id is not None else "updated"


def get_trial(db: Database, trial_id: str) -> Optional[dict]:
    collection = db[CLINICAL_TRIALS_COLLECTION]
    return collection.find_one({"trial_id": trial_id}, _EXCLUDE_MONGO_ID)


def list_trials(
    db: Database,
    page: int,
    page_size: int,
    status: Optional[str] = None,
    condition: Optional[str] = None,
) -> tuple[list[dict], int]:
    collection = db[CLINICAL_TRIALS_COLLECTION]
    query: dict[str, Any] = {}
    if status:
        query["status"] = status
    if condition:
        query["conditions.name"] = condition

    skip = (page - 1) * page_size
    total = collection.count_documents(query)
    cursor = collection.find(query, _EXCLUDE_MONGO_ID).sort("trial_id", 1).skip(skip).limit(page_size)
    return list(cursor), total


def iter_candidate_trials(db: Database, status: Optional[str] = None) -> Iterator[dict]:
    """Lazily yields every trial matching the given status, one at a time.

    Used by the matching engine, which must evaluate the complete candidate
    set but must not hold the entire clinical_trials collection in memory
    at once. The query filters on `status`, which is covered by the
    `idx_trial_status` index (see db/mongodb.py) — MongoDB serves matching
    documents to the cursor in batches rather than the driver ever
    materializing the full result set up front. Callers should iterate this
    (e.g. in a list/generator comprehension) rather than calling list() on
    it, or the memory-usage benefit is lost.
    """
    collection = db[CLINICAL_TRIALS_COLLECTION]
    query: dict[str, Any] = {"status": status} if status else {}
    for document in collection.find(query, _EXCLUDE_MONGO_ID):
        yield document


def count_trials(db: Database) -> int:
    return db[CLINICAL_TRIALS_COLLECTION].count_documents({})
