"""Ingestion of the local synthetic/development clinical trial fixture into
MongoDB.

SOURCE RESTRICTION: this service reads exclusively from the local fixture
file configured by TRIALS_DATA_PATH (data/trials/dev_trials.json by
default). It never accepts trial data from a request body or any other
source, and it refuses to ingest a fixture file that is missing the
required synthetic-data disclaimer — trials stored by this service can
never be mistaken for real, registered clinical trials.

Run locally with:
    python -m app.services.trial_ingestion
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError
from pymongo.database import Database

from app.config import get_settings
from app.db.mongodb import ensure_indexes, get_database
from app.models.clinical_trial import ClinicalTrialOut
from app.repositories import trial_repository

logger = logging.getLogger(__name__)

_DISCLAIMER_KEY = "_disclaimer"


class TrialFixtureError(Exception):
    """The trials fixture file itself is structurally invalid (missing,
    malformed JSON, or missing the required synthetic-data disclaimer) —
    raised instead of silently ingesting nothing or guessing at its shape."""


@dataclass
class TrialIngestionStats:
    trials_discovered: int = 0
    trials_inserted: int = 0
    trials_updated: int = 0
    trials_failed: int = 0
    errors: list[str] = field(default_factory=list)


class TrialIngestionService:
    """Loads the local synthetic trials fixture, validates/normalizes each
    record, and upserts it into MongoDB keyed on trial_id. Safe to run
    repeatedly — never creates duplicates."""

    def __init__(self, db: Database, trials_path: Optional[Path] = None):
        self.db = db
        settings = get_settings()
        self.trials_path = Path(trials_path) if trials_path is not None else Path(settings.trials_data_path)

    def load_trial_records(self) -> list[dict[str, Any]]:
        if not self.trials_path.exists():
            raise TrialFixtureError(f"Trials fixture not found: {self.trials_path}")

        try:
            raw_text = self.trials_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise TrialFixtureError(f"Could not read trials fixture {self.trials_path}: {exc}") from exc

        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise TrialFixtureError(f"Invalid JSON in trials fixture {self.trials_path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise TrialFixtureError(
                f"Trials fixture {self.trials_path} must be a JSON object with "
                f"'{_DISCLAIMER_KEY}' and 'trials' keys"
            )

        disclaimer = raw.get(_DISCLAIMER_KEY)
        if not isinstance(disclaimer, str) or "synthetic" not in disclaimer.lower():
            raise TrialFixtureError(
                f"Trials fixture {self.trials_path} is missing the required synthetic/development-data "
                f"disclaimer ('{_DISCLAIMER_KEY}') — refusing to ingest data not clearly marked as fictional"
            )

        records = raw.get("trials")
        if not isinstance(records, list):
            raise TrialFixtureError(f"Trials fixture {self.trials_path} has no valid 'trials' list")

        return records

    @staticmethod
    def _normalize_record(record: Any) -> ClinicalTrialOut:
        if not isinstance(record, dict):
            raise ValueError("trial record is not a JSON object")
        if not record.get("trial_id"):
            raise ValueError("trial record is missing required 'trial_id'")

        document = dict(record)
        # Provenance is always this local synthetic fixture — the record's
        # own claimed source/source_id is never trusted, so a synthetic
        # trial can never masquerade as coming from elsewhere.
        document["source"] = "synthetic_dev_fixture"
        document["source_id"] = document.get("source_id") or document["trial_id"]
        document["ingested_at"] = datetime.now(timezone.utc)

        return ClinicalTrialOut(**document)

    def run(self) -> TrialIngestionStats:
        stats = TrialIngestionStats()

        try:
            records = self.load_trial_records()
        except TrialFixtureError as exc:
            stats.errors.append(str(exc))
            return stats

        stats.trials_discovered = len(records)

        for record in records:
            try:
                trial = self._normalize_record(record)
            except (ValueError, ValidationError) as exc:
                stats.trials_failed += 1
                trial_id = record.get("trial_id", "<unknown>") if isinstance(record, dict) else "<unknown>"
                stats.errors.append(f"{trial_id}: {exc}")
                continue

            result = trial_repository.upsert_trial(self.db, trial.model_dump())
            if result == "inserted":
                stats.trials_inserted += 1
            else:
                stats.trials_updated += 1

        return stats


def run_ingestion() -> TrialIngestionStats:
    db = get_database()
    ensure_indexes(db)
    return TrialIngestionService(db).run()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("Trial ingestion started")
    print("NOTE: ingesting SYNTHETIC/DEVELOPMENT trial fixtures only — these are not real clinical trials.")
    stats = run_ingestion()

    print()
    print(f"Trials discovered: {stats.trials_discovered}")
    print(f"Inserted: {stats.trials_inserted}")
    print(f"Updated: {stats.trials_updated}")
    print(f"Failed: {stats.trials_failed}")

    if stats.errors:
        print()
        print(f"Errors ({len(stats.errors)}):")
        for error in stats.errors[:50]:
            print(f"  - {error}")
        if len(stats.errors) > 50:
            print(f"  ... and {len(stats.errors) - 50} more")

    print()
    print("Trial ingestion completed")


if __name__ == "__main__":
    main()
