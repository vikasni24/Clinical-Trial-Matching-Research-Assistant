import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo.errors import PyMongoError

from app.api.routes import fhir, patients, trials
from app.config import get_settings
from app.db.mongodb import ensure_indexes, get_database

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ensure_indexes(get_database())
    except PyMongoError as exc:
        logger.warning("Could not ensure MongoDB indexes at startup: %s", exc)
    yield


app = FastAPI(
    title="Clinical Trial Matching & Research Assistant - Phase 1",
    description="FHIR ingestion, storage, and retrieval APIs.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(patients.router)
app.include_router(fhir.router)
app.include_router(trials.router)

# Frontend Phase 1: allow the local frontend dev server to call this API.
# Origins are configurable (see Settings.cors_allowed_origins) and default
# to Vite's own dev server ports only — no wildcard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(PyMongoError)
async def mongo_error_handler(request: Request, exc: PyMongoError) -> JSONResponse:
    logger.error("MongoDB error on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=503, content={"detail": "Database temporarily unavailable"})


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}
