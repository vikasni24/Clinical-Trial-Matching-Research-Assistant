"""MongoDB connection management and index setup."""

from functools import lru_cache

from pymongo import MongoClient
from pymongo.database import Database

from app.config import get_settings

FHIR_RESOURCES_COLLECTION = "fhir_resources"
PATIENTS_COLLECTION = "patients"
PATIENT_PROFILES_COLLECTION = "patient_profiles"
CLINICAL_TRIALS_COLLECTION = "clinical_trials"


@lru_cache
def get_client() -> MongoClient:
    settings = get_settings()
    return MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=5000)


def get_database() -> Database:
    """FastAPI-dependency-friendly accessor for the configured database."""
    settings = get_settings()
    return get_client()[settings.mongodb_database]


def ensure_indexes(db: Database) -> None:
    """Create indexes needed for resource_type / resource_id / patient_id lookups
    and safe upsert-based deduplication, sized for large (1M+) collections."""
    fhir_resources = db[FHIR_RESOURCES_COLLECTION]
    fhir_resources.create_index(
        [("resource_type", 1), ("resource_id", 1)],
        unique=True,
        name="uniq_resource_type_id",
    )
    fhir_resources.create_index("patient_id", name="idx_patient_id")
    fhir_resources.create_index(
        [("patient_id", 1), ("resource_type", 1)],
        name="idx_patient_resource_type",
    )

    patients = db[PATIENTS_COLLECTION]
    patients.create_index("patient_id", unique=True, name="uniq_patient_id")

    patient_profiles = db[PATIENT_PROFILES_COLLECTION]
    patient_profiles.create_index("patient_id", unique=True, name="uniq_profile_patient_id")

    clinical_trials = db[CLINICAL_TRIALS_COLLECTION]
    clinical_trials.create_index("trial_id", unique=True, name="uniq_trial_id")
    clinical_trials.create_index("status", name="idx_trial_status")
    clinical_trials.create_index("conditions.name", name="idx_trial_conditions")
