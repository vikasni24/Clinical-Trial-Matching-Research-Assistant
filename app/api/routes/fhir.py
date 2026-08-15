from fastapi import APIRouter, Depends, HTTPException
from pymongo.database import Database

from app.db.mongodb import get_database
from app.models.schemas import FHIRResourceOut
from app.repositories import fhir_repository

router = APIRouter(prefix="/api/fhir", tags=["fhir"])


@router.get("/{resource_type}/{resource_id}", response_model=FHIRResourceOut)
def get_fhir_resource(
    resource_type: str, resource_id: str, db: Database = Depends(get_database)
) -> FHIRResourceOut:
    resource = fhir_repository.get_fhir_resource(db, resource_type=resource_type, resource_id=resource_id)
    if resource is None:
        raise HTTPException(status_code=404, detail=f"{resource_type}/{resource_id} not found")
    return FHIRResourceOut(**resource)
