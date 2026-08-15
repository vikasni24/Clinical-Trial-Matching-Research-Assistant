"""Read-only trial browsing plus a manual ingestion trigger.

All trial data served here comes exclusively from the local synthetic
fixture configured by TRIALS_DATA_PATH (see TrialIngestionService) — never
from a request body or any external source. Every list/ingest response
carries an explicit disclaimer identifying the data as synthetic/
development-only; individual trial records also carry this in their
`source`/`sponsor` fields.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pymongo.database import Database

from app.config import get_settings
from app.db.mongodb import get_database
from app.models.clinical_trial import ClinicalTrialOut, TrialIngestionReportOut, TrialListOut
from app.models.schemas import PaginationMeta
from app.repositories import trial_repository
from app.services.trial_ingestion import TrialIngestionService

router = APIRouter(prefix="/api/trials", tags=["trials"])


def _resolve_page_size(page_size: Optional[int]) -> int:
    settings = get_settings()
    size = page_size or settings.default_page_size
    return min(size, settings.max_page_size)


def _total_pages(total: int, page_size: int) -> int:
    return (total + page_size - 1) // page_size if total else 0


@router.get("", response_model=TrialListOut)
def list_trials(
    page: int = Query(1, ge=1),
    page_size: Optional[int] = Query(None, ge=1),
    status: Optional[str] = Query(None),
    condition: Optional[str] = Query(None),
    db: Database = Depends(get_database),
) -> TrialListOut:
    size = _resolve_page_size(page_size)
    items, total = trial_repository.list_trials(db, page=page, page_size=size, status=status, condition=condition)
    return TrialListOut(
        items=[ClinicalTrialOut(**item) for item in items],
        pagination=PaginationMeta(page=page, page_size=size, total=total, total_pages=_total_pages(total, size)),
    )


@router.get("/{trial_id}", response_model=ClinicalTrialOut)
def get_trial(trial_id: str, db: Database = Depends(get_database)) -> ClinicalTrialOut:
    trial = trial_repository.get_trial(db, trial_id)
    if trial is None:
        raise HTTPException(status_code=404, detail=f"Trial '{trial_id}' not found")
    return ClinicalTrialOut(**trial)


@router.post("/ingest", response_model=TrialIngestionReportOut)
def ingest_trials(db: Database = Depends(get_database)) -> TrialIngestionReportOut:
    """Re-reads the local synthetic trials fixture and upserts it into
    MongoDB. Accepts no request body — it can only ever ingest the
    configured local fixture file, never caller-supplied trial data."""
    stats = TrialIngestionService(db).run()
    return TrialIngestionReportOut(
        trials_discovered=stats.trials_discovered,
        trials_inserted=stats.trials_inserted,
        trials_updated=stats.trials_updated,
        trials_failed=stats.trials_failed,
        errors=stats.errors,
    )
