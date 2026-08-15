"""Data access layer for FHIR resources and patients stored in MongoDB.

Keeps all query/index-shape knowledge in one place so callers (API routes,
the ingestion service) never build ad hoc queries against the collections.
"""

from typing import Any, Optional

from pymongo.database import Database

from app.db.mongodb import FHIR_RESOURCES_COLLECTION, PATIENTS_COLLECTION

_EXCLUDE_MONGO_ID = {"_id": 0}


def upsert_fhir_resource(db: Database, document: dict[str, Any]) -> str:
    """Insert or update a FHIR resource keyed on (resource_type, resource_id).

    Returns "inserted" or "updated".
    """
    collection = db[FHIR_RESOURCES_COLLECTION]
    filter_query = {
        "resource_type": document["resource_type"],
        "resource_id": document["resource_id"],
    }
    result = collection.update_one(filter_query, {"$set": document}, upsert=True)
    return "inserted" if result.upserted_id is not None else "updated"


def upsert_patient(db: Database, document: dict[str, Any]) -> str:
    """Insert or update a patient keyed on patient_id. Returns "inserted" or "updated"."""
    collection = db[PATIENTS_COLLECTION]
    filter_query = {"patient_id": document["patient_id"]}
    result = collection.update_one(filter_query, {"$set": document}, upsert=True)
    return "inserted" if result.upserted_id is not None else "updated"


def list_patients(db: Database, page: int, page_size: int) -> tuple[list[dict], int]:
    collection = db[PATIENTS_COLLECTION]
    skip = (page - 1) * page_size
    total = collection.count_documents({})
    cursor = (
        collection.find({}, _EXCLUDE_MONGO_ID)
        .sort("patient_id", 1)
        .skip(skip)
        .limit(page_size)
    )
    return list(cursor), total


def get_patient(db: Database, patient_id: str) -> Optional[dict]:
    collection = db[PATIENTS_COLLECTION]
    return collection.find_one({"patient_id": patient_id}, _EXCLUDE_MONGO_ID)


def list_patient_resources(
    db: Database,
    patient_id: str,
    resource_type: Optional[str],
    page: int,
    page_size: int,
) -> tuple[list[dict], int]:
    collection = db[FHIR_RESOURCES_COLLECTION]
    query: dict[str, Any] = {"patient_id": patient_id}
    if resource_type:
        query["resource_type"] = resource_type

    skip = (page - 1) * page_size
    total = collection.count_documents(query)
    cursor = (
        collection.find(query, _EXCLUDE_MONGO_ID)
        .sort([("resource_type", 1), ("resource_id", 1)])
        .skip(skip)
        .limit(page_size)
    )
    return list(cursor), total


def get_fhir_resource(db: Database, resource_type: str, resource_id: str) -> Optional[dict]:
    collection = db[FHIR_RESOURCES_COLLECTION]
    return collection.find_one(
        {"resource_type": resource_type, "resource_id": resource_id},
        _EXCLUDE_MONGO_ID,
    )


def list_all_patient_ids(db: Database) -> list[str]:
    """All known patient ids, for batch jobs (e.g. normalization) that must
    visit every patient rather than a single page."""
    collection = db[PATIENTS_COLLECTION]
    cursor = collection.find({}, {"_id": 0, "patient_id": 1}).sort("patient_id", 1)
    return [doc["patient_id"] for doc in cursor]


def get_all_patient_resources(db: Database, patient_id: str) -> list[dict]:
    """Every FHIR resource for a single patient, unpaginated. Used by the
    normalization service, which must see the complete resource set rather
    than a page of it."""
    collection = db[FHIR_RESOURCES_COLLECTION]
    return list(collection.find({"patient_id": patient_id}, _EXCLUDE_MONGO_ID))
