from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pymongo.database import Database

from app.config import get_settings
from app.db.mongodb import get_database
from app.models.match_result import MatchListOut, MatchResultOut
from app.models.patient_profile import PatientProfileOut
from app.models.schemas import (
    FHIRResourceOut,
    PaginationMeta,
    PatientListOut,
    PatientOut,
    ResourceListOut,
)
from app.repositories import fhir_repository, patient_profile_repository, trial_repository
from app.services.trial_matching import TrialMatchingService

router = APIRouter(prefix="/api/patients", tags=["patients"])


def _resolve_page_size(page_size: Optional[int]) -> int:
    settings = get_settings()
    size = page_size or settings.default_page_size
    return min(size, settings.max_page_size)


def _total_pages(total: int, page_size: int) -> int:
    return (total + page_size - 1) // page_size if total else 0


@router.get("", response_model=PatientListOut)
def list_patients(
    page: int = Query(1, ge=1),
    page_size: Optional[int] = Query(None, ge=1),
    db: Database = Depends(get_database),
) -> PatientListOut:
    size = _resolve_page_size(page_size)
    items, total = fhir_repository.list_patients(db, page=page, page_size=size)
    return PatientListOut(
        items=[PatientOut(**item) for item in items],
        pagination=PaginationMeta(page=page, page_size=size, total=total, total_pages=_total_pages(total, size)),
    )


@router.get("/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: str, db: Database = Depends(get_database)) -> PatientOut:
    patient = fhir_repository.get_patient(db, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")
    return PatientOut(**patient)


@router.get("/{patient_id}/resources", response_model=ResourceListOut)
def get_patient_resources(
    patient_id: str,
    resource_type: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: Optional[int] = Query(None, ge=1),
    db: Database = Depends(get_database),
) -> ResourceListOut:
    patient = fhir_repository.get_patient(db, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")

    size = _resolve_page_size(page_size)
    items, total = fhir_repository.list_patient_resources(
        db, patient_id=patient_id, resource_type=resource_type, page=page, page_size=size
    )
    return ResourceListOut(
        items=[FHIRResourceOut(**item) for item in items],
        pagination=PaginationMeta(page=page, page_size=size, total=total, total_pages=_total_pages(total, size)),
    )


@router.get("/{patient_id}/profile", response_model=PatientProfileOut)
def get_patient_profile(patient_id: str, db: Database = Depends(get_database)) -> PatientProfileOut:
    profile = patient_profile_repository.get_patient_profile(db, patient_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Patient profile for '{patient_id}' not found")
    return PatientProfileOut(**profile)


@router.get("/{patient_id}/matches", response_model=MatchListOut)
def get_patient_matches(
    patient_id: str,
    status: Optional[str] = Query("recruiting"),
    db: Database = Depends(get_database),
) -> MatchListOut:
    """Deterministically evaluates the patient against candidate trials.
    By default only "recruiting" trials are considered (pass status=all to
    evaluate every trial regardless of recruitment status, or another
    status value to filter to it)."""
    if patient_profile_repository.get_patient_profile(db, patient_id) is None:
        raise HTTPException(status_code=404, detail=f"Patient profile for '{patient_id}' not found")

    status_filter = None if status and status.lower() == "all" else status
    results = TrialMatchingService(db).match_patient_to_trials(patient_id, status=status_filter)
    return MatchListOut(patient_id=patient_id, total_trials_evaluated=len(results), matches=results)


@router.get("/{patient_id}/matches/{trial_id}", response_model=MatchResultOut)
def get_patient_trial_match(patient_id: str, trial_id: str, db: Database = Depends(get_database)) -> MatchResultOut:
    if patient_profile_repository.get_patient_profile(db, patient_id) is None:
        raise HTTPException(status_code=404, detail=f"Patient profile for '{patient_id}' not found")
    if trial_repository.get_trial(db, trial_id) is None:
        raise HTTPException(status_code=404, detail=f"Trial '{trial_id}' not found")

    return TrialMatchingService(db).match_patient_to_trial(patient_id, trial_id)
