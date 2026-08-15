"""Data access layer for normalized PatientProfile documents.

Mirrors the conventions in fhir_repository.py: all query/index-shape
knowledge for the patient_profiles collection lives here.
"""

from typing import Any, Optional

from pymongo.database import Database

from app.db.mongodb import PATIENT_PROFILES_COLLECTION

_EXCLUDE_MONGO_ID = {"_id": 0}


def upsert_patient_profile(db: Database, document: dict[str, Any]) -> str:
    """Insert or update a PatientProfile keyed on patient_id.

    Returns "inserted" or "updated".
    """
    collection = db[PATIENT_PROFILES_COLLECTION]
    filter_query = {"patient_id": document["patient_id"]}
    result = collection.update_one(filter_query, {"$set": document}, upsert=True)
    return "inserted" if result.upserted_id is not None else "updated"


def get_patient_profile(db: Database, patient_id: str) -> Optional[dict]:
    collection = db[PATIENT_PROFILES_COLLECTION]
    return collection.find_one({"patient_id": patient_id}, _EXCLUDE_MONGO_ID)


def count_patient_profiles(db: Database) -> int:
    return db[PATIENT_PROFILES_COLLECTION].count_documents({})
