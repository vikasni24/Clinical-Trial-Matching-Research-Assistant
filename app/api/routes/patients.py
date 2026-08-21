import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pymongo.database import Database

from app.config import get_settings
from app.db.mongodb import get_database
from app.models.answer import AskRequest, GroundedAnswer
from app.models.audit import AuditHistoryOut, AuditRecordOut
from app.models.evidence import EvidenceListOut
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
from app.services.anthropic_llm_provider import (
    AnthropicLLMProvider,
    LLMProviderConfigurationError,
    LLMProviderRequestError,
)
from app.services.ask_service import AskService
from app.services.audit_service import AuditService
from app.services.evidence_service import EvidenceService
from app.services.groq_llm_provider import GroqLLMProvider
from app.services.llm_provider import LLMProvider
from app.services.trial_matching import TrialMatchingService

router = APIRouter(prefix="/api/patients", tags=["patients"])
logger = logging.getLogger(__name__)


class _LazyConfiguredLLMProvider:
    """Defers constructing the real, configured LLM provider — and
    therefore its LLM_API_KEY configuration check — until generate() is
    actually called. Without this, resolving the get_llm_provider
    dependency below would fail every request (even ones that never need
    generation, e.g. an unknown patient or an empty query) whenever no API
    key is configured. Which concrete provider gets constructed is
    controlled entirely by Settings.llm_provider ("anthropic" or "groq")
    — AskService itself never knows or cares which one it was given."""

    def generate(self, prompt: str) -> str:
        settings = get_settings()
        provider_name = (settings.llm_provider or "anthropic").strip().lower()
        if provider_name == "anthropic":
            return AnthropicLLMProvider(settings=settings).generate(prompt)
        if provider_name == "groq":
            return GroqLLMProvider(settings=settings).generate(prompt)
        raise LLMProviderConfigurationError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}' — supported values are 'anthropic' and 'groq'"
        )


def get_llm_provider() -> LLMProvider:
    """FastAPI-dependency-friendly accessor for the configured LLM
    provider. Overridden in tests (see tests/test_api_ask.py) with a
    deterministic fake — a real LLM is never called from automated tests."""
    return _LazyConfiguredLLMProvider()


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


@router.get("/{patient_id}/evidence", response_model=EvidenceListOut)
def get_patient_evidence(
    patient_id: str,
    resource_type: Optional[str] = Query(None),
    code: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: Optional[int] = Query(None, ge=1),
    db: Database = Depends(get_database),
) -> EvidenceListOut:
    """Patient-scoped, lightweight clinical evidence derived from the
    patient's FHIR resources (see app/models/evidence.py) — never the raw
    FHIR document; use GET /api/fhir/{resource_type}/{resource_id} for
    that. Distinct from GET /matches/{trial_id}'s evidence (which is
    scoped to whatever a specific trial's criteria reference): this
    endpoint supports general-purpose browsing of a patient's evidence,
    e.g. by a future research-assistant UI."""
    if fhir_repository.get_patient(db, patient_id) is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")

    size = _resolve_page_size(page_size)
    items, total = EvidenceService(db).list_patient_evidence(
        patient_id, resource_type=resource_type, code=code, page=page, page_size=size
    )
    return EvidenceListOut(
        items=items,
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


@router.post("/{patient_id}/ask", response_model=GroundedAnswer)
def ask_patient_question(
    patient_id: str,
    payload: AskRequest,
    db: Database = Depends(get_database),
    llm_provider: LLMProvider = Depends(get_llm_provider),
) -> GroundedAnswer:
    """The grounded question-answering pipeline (Phase 6G): patient_id ->
    HybridEvidenceRetriever -> GroundedContext -> grounded prompt builder ->
    LLMProvider -> answer validation -> GroundedAnswer. Never retrieves
    another patient's data, never bypasses the retrieval layer, never sends
    raw FHIR to the LLM, never invents evidence, and never treats missing
    evidence as a negative clinical fact — all of that is enforced,
    unmodified, by AskService's own dependencies (Phases 5E, 6B-6F)."""
    if fhir_repository.get_patient(db, patient_id) is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")

    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must not be empty")

    try:
        return AskService(db, llm_provider).ask(patient_id, query)
    except LLMProviderConfigurationError as exc:
        # Always a static, secret-free message (e.g. "LLM_API_KEY is not
        # configured") — safe to return to the client verbatim.
        raise HTTPException(status_code=500, detail=f"LLM provider is not configured: {exc}") from exc
    except LLMProviderRequestError as exc:
        # exc may embed the raw upstream HTTP response body (see
        # AnthropicLLMProvider.generate) — that must never reach the API
        # client. Full detail is logged server-side only; the client gets a
        # generic, safe message.
        logger.warning("LLM provider request failed for patient '%s': %s", patient_id, exc)
        raise HTTPException(status_code=502, detail="LLM provider request failed") from exc


@router.get("/{patient_id}/audit", response_model=AuditHistoryOut)
def get_patient_audit_history(
    patient_id: str,
    page: int = Query(1, ge=1),
    page_size: Optional[int] = Query(None, ge=1),
    db: Database = Depends(get_database),
) -> AuditHistoryOut:
    """The deterministic audit trail for this patient's grounded questions
    (Phase 7): newest first, patient-scoped, paginated. Never exposes raw
    FHIR, raw LLM output, prompts, secrets, or hidden reasoning — only the
    facts already established by the /ask pipeline (see
    app/services/audit_service.py, app/models/audit.py)."""
    if fhir_repository.get_patient(db, patient_id) is None:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found")

    size = _resolve_page_size(page_size)
    records, total = AuditService(db).get_patient_audit_history(patient_id, page=page, page_size=size)
    return AuditHistoryOut(
        items=[AuditRecordOut.from_record(record) for record in records],
        pagination=PaginationMeta(page=page, page_size=size, total=total, total_pages=_total_pages(total, size)),
    )
