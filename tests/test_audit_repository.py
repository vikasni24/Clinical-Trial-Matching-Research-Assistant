"""Phase 7C: audit_repository — insertion and patient-scoped, paginated
retrieval against the `audit_records` collection. Uses mongomock; no live
MongoDB required."""

from datetime import datetime, timedelta, timezone

from app.repositories import audit_repository
from app.models.audit import AuditRecord


def _record(audit_id, patient_id="p1", created_at=None, **overrides):
    defaults = dict(
        audit_id=audit_id,
        patient_id=patient_id,
        query="hypertension?",
        retrieval_status="evidence_found",
        answer_status="answered",
        evidence_references=[],
        created_at=created_at or datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    return AuditRecord(**defaults)


# --- 5: insert ---------------------------------------------------------------------------


def test_insert_audit_record(mongo_db):
    record = _record("audit-1")

    audit_repository.insert_audit_record(mongo_db, record)

    stored = mongo_db["audit_records"].find_one({"audit_id": "audit-1"})
    assert stored is not None
    assert stored["patient_id"] == "p1"


def test_insert_never_persists_a_mongo_id_back_onto_the_record(mongo_db):
    record = _record("audit-2")

    audit_repository.insert_audit_record(mongo_db, record)

    # The AuditRecord object itself is untouched by the insert.
    assert not hasattr(record, "_id")


# --- 6: patient-scoped retrieval ----------------------------------------------------------


def test_patient_scoped_audit_retrieval(mongo_db):
    audit_repository.insert_audit_record(mongo_db, _record("audit-1", patient_id="p1"))
    audit_repository.insert_audit_record(mongo_db, _record("audit-2", patient_id="p1"))

    items, total = audit_repository.get_patient_audit_history(mongo_db, "p1")

    assert total == 2
    assert {item.audit_id for item in items} == {"audit-1", "audit-2"}
    assert all(item.patient_id == "p1" for item in items)


# --- 7: cross-patient audit isolation -----------------------------------------------------


def test_cross_patient_audit_isolation(mongo_db):
    audit_repository.insert_audit_record(mongo_db, _record("audit-a", patient_id="patient-a"))
    audit_repository.insert_audit_record(mongo_db, _record("audit-b", patient_id="patient-b"))

    items_a, total_a = audit_repository.get_patient_audit_history(mongo_db, "patient-a")
    items_b, total_b = audit_repository.get_patient_audit_history(mongo_db, "patient-b")

    assert total_a == 1 and [i.audit_id for i in items_a] == ["audit-a"]
    assert total_b == 1 and [i.audit_id for i in items_b] == ["audit-b"]


def test_unknown_patient_returns_empty_history_not_an_error(mongo_db):
    audit_repository.insert_audit_record(mongo_db, _record("audit-1", patient_id="p1"))

    items, total = audit_repository.get_patient_audit_history(mongo_db, "does-not-exist")

    assert items == []
    assert total == 0


# --- 8: pagination -------------------------------------------------------------------------


def test_pagination(mongo_db):
    base_time = datetime.now(timezone.utc)
    for i in range(5):
        audit_repository.insert_audit_record(
            mongo_db, _record(f"audit-{i}", patient_id="p1", created_at=base_time + timedelta(seconds=i))
        )

    page_1, total = audit_repository.get_patient_audit_history(mongo_db, "p1", page=1, page_size=2)
    page_2, _ = audit_repository.get_patient_audit_history(mongo_db, "p1", page=2, page_size=2)

    assert total == 5
    assert len(page_1) == 2
    assert len(page_2) == 2
    assert {r.audit_id for r in page_1}.isdisjoint({r.audit_id for r in page_2})


def test_pagination_never_loads_more_than_page_size_documents(mongo_db):
    base_time = datetime.now(timezone.utc)
    for i in range(50):
        audit_repository.insert_audit_record(
            mongo_db, _record(f"audit-{i}", patient_id="p1", created_at=base_time + timedelta(seconds=i))
        )

    items, total = audit_repository.get_patient_audit_history(mongo_db, "p1", page=1, page_size=10)

    assert total == 50
    assert len(items) == 10  # never the full 50-record history


# --- 9: newest-first ordering ----------------------------------------------------------------


def test_newest_first_ordering(mongo_db):
    base_time = datetime.now(timezone.utc)
    audit_repository.insert_audit_record(mongo_db, _record("oldest", patient_id="p1", created_at=base_time))
    audit_repository.insert_audit_record(
        mongo_db, _record("middle", patient_id="p1", created_at=base_time + timedelta(seconds=10))
    )
    audit_repository.insert_audit_record(
        mongo_db, _record("newest", patient_id="p1", created_at=base_time + timedelta(seconds=20))
    )

    items, _ = audit_repository.get_patient_audit_history(mongo_db, "p1")

    assert [item.audit_id for item in items] == ["newest", "middle", "oldest"]


# --- never an unscoped query -----------------------------------------------------------------


def test_repository_never_issues_an_unscoped_query(mongo_db, monkeypatch):
    original_find = mongo_db["audit_records"].find
    captured_queries = []

    def spy_find(query, *args, **kwargs):
        captured_queries.append(query)
        return original_find(query, *args, **kwargs)

    monkeypatch.setattr(mongo_db["audit_records"], "find", spy_find)
    audit_repository.insert_audit_record(mongo_db, _record("audit-1", patient_id="p1"))

    audit_repository.get_patient_audit_history(mongo_db, "p1")

    assert len(captured_queries) == 1
    assert captured_queries[0] == {"patient_id": "p1"}
    assert captured_queries[0] != {}
