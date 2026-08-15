"""FHIR ingestion service.

Recursively discovers Synthea-generated FHIR JSON files (standalone
resources or Bundles) and loads supported resource types into MongoDB,
preserving the original FHIR JSON and provenance metadata for every
stored resource. Safe to run repeatedly (upsert on resource_type + resource_id).

Run locally with:
    python -m app.services.fhir_ingestion
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pymongo.database import Database

from app.config import get_settings
from app.db.mongodb import ensure_indexes, get_database
from app.repositories.fhir_repository import upsert_fhir_resource, upsert_patient
from app.services.fhir_parser import extract_resources

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    files_discovered: int = 0
    files_processed: int = 0
    files_failed: int = 0
    resources_processed: int = 0
    resources_inserted: int = 0
    resources_updated: int = 0
    resources_skipped: int = 0
    resources_failed: int = 0
    errors: list[str] = field(default_factory=list)


class FHIRIngestionService:
    """Discovers, parses, and persists FHIR resources from a local directory."""

    def __init__(self, db: Database, fhir_dir: Optional[Path] = None):
        self.db = db
        settings = get_settings()
        self.fhir_dir = Path(fhir_dir) if fhir_dir is not None else Path(settings.synthea_fhir_dir)

    def discover_files(self) -> list[Path]:
        if not self.fhir_dir.exists():
            return []
        return sorted(p for p in self.fhir_dir.rglob("*.json") if p.is_file())

    def ingest_file(self, path: Path, stats: IngestionStats) -> None:
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            stats.files_failed += 1
            stats.errors.append(f"{path}: could not read file ({exc})")
            return

        try:
            raw = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            stats.files_failed += 1
            stats.errors.append(f"{path}: invalid JSON ({exc})")
            return

        stats.files_processed += 1
        outcomes = extract_resources(raw, source_file=path.name)

        for outcome in outcomes:
            if outcome.status == "skipped_unsupported":
                stats.resources_skipped += 1
                continue
            if outcome.status == "failed":
                stats.resources_failed += 1
                if outcome.message:
                    stats.errors.append(outcome.message)
                continue

            stats.resources_processed += 1
            resource = outcome.resource
            document = {
                "resource_type": resource.resource_type,
                "resource_id": resource.resource_id,
                "patient_id": resource.patient_id,
                "source_file": path.name,
                "source_bundle_id": resource.source_bundle_id,
                "ingested_at": datetime.now(timezone.utc),
                "data": resource.data,
            }
            try:
                result = upsert_fhir_resource(self.db, document)
                if result == "inserted":
                    stats.resources_inserted += 1
                else:
                    stats.resources_updated += 1

                if resource.resource_type == "Patient":
                    upsert_patient(
                        self.db,
                        {
                            "patient_id": resource.resource_id,
                            "source_file": path.name,
                            "ingested_at": document["ingested_at"],
                            "data": resource.data,
                        },
                    )
            except Exception as exc:  # e.g. pymongo errors
                stats.resources_failed += 1
                stats.errors.append(
                    f"{path}: failed to store {resource.resource_type}/{resource.resource_id} ({exc})"
                )

    def run(self) -> IngestionStats:
        stats = IngestionStats()
        files = self.discover_files()
        stats.files_discovered = len(files)

        for path in files:
            self.ingest_file(path, stats)

        return stats


def run_ingestion() -> IngestionStats:
    db = get_database()
    ensure_indexes(db)
    return FHIRIngestionService(db).run()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("FHIR ingestion started")
    stats = run_ingestion()

    print()
    print(f"Files discovered: {stats.files_discovered}")
    print(f"Resources processed: {stats.resources_processed}")
    print(f"Inserted: {stats.resources_inserted}")
    print(f"Updated: {stats.resources_updated}")
    print(f"Skipped: {stats.resources_skipped}")
    print(f"Failed: {stats.resources_failed}")

    if stats.errors:
        print()
        print(f"Errors ({len(stats.errors)}):")
        for error in stats.errors[:50]:
            print(f"  - {error}")
        if len(stats.errors) > 50:
            print(f"  ... and {len(stats.errors) - 50} more")

    print()
    print("FHIR ingestion completed")


if __name__ == "__main__":
    main()
