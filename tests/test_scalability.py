"""Phase 8: scalability/performance-hardening verification.

This phase's inspection (8A-8J) found every patient-scoped query already
scoped at the MongoDB layer, every list-returning API endpoint already
MongoDB-side paginated, and both lazy generators (iter_candidate_trials,
evidence_repository.get_patient_evidence*) never eagerly materialized in
their own module. No production code changes were required — this file
exists to make those guarantees explicit, regression-tested claims rather
than an unverified inspection note. No real LLM is ever called."""

from __future__ import annotations

import inspect
import shutil
from datetime import datetime, timezone

from app.api.routes.patients import get_llm_provider
from app.db.mongodb import ensure_indexes
from app.main import app
from app.models.audit import AuditRecord
from app.models.clinical_trial import ClinicalTrialOut, EligibilityCriterionOut, TrialEligibilityOut
from app.models.patient_profile import ContactInfoOut, DemographicsOut, PatientProfileOut
from app.repositories import audit_repository, evidence_repository, fhir_repository, trial_repository
from app.services.eligibility_matcher import EligibilityMatcher
from app.services.evidence_service import EvidenceService
from app.services.fhir_ingestion import FHIRIngestionService
from app.services.hybrid_retriever import DEFAULT_LIMIT as HYBRID_DEFAULT_LIMIT
from app.services.hybrid_retriever import HybridEvidenceRetriever
from app.services.patient_normalization import PatientNormalizationService
from app.models.retrieval import RetrievalRequest


def _setup(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()
    PatientNormalizationService(mongo_db).normalize_all()


def _insert_observations(mongo_db, patient_id, count, code="8480-6", display="Systolic Blood Pressure"):
    for i in range(count):
        mongo_db["fhir_resources"].insert_one(
            {
                "patient_id": patient_id,
                "resource_type": "Observation",
                "resource_id": f"bulk-obs-{i}",
                "data": {
                    "resourceType": "Observation",
                    "id": f"bulk-obs-{i}",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": display}], "text": display},
                    "effectiveDateTime": f"2023-01-{(i % 28) + 1:02d}",
                    "valueQuantity": {"value": 100 + i, "unit": "mm[Hg]"},
                },
            }
        )


class _FakeLLMProvider:
    def __init__(self, response: str = "unused"):
        self.calls: list[str] = []
        self._response = response

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self._response


def _override_llm(fake):
    app.dependency_overrides[get_llm_provider] = lambda: fake


def _clear_llm_override():
    app.dependency_overrides.pop(get_llm_provider, None)


# --- 1: trial candidate retrieval remains lazy --------------------------------------------


def test_iter_candidate_trials_is_a_generator_function():
    assert inspect.isgeneratorfunction(trial_repository.iter_candidate_trials)


def test_iter_candidate_trials_is_never_materialized_inside_trial_matching(mongo_db):
    import app.services.trial_matching as module

    source = inspect.getsource(module)
    assert "list(trial_repository.iter_candidate_trials" not in source
    assert "list(" + "iter_candidate_trials" not in source


# --- 2: evidence retrieval remains lazy -----------------------------------------------------


def test_get_patient_evidence_is_a_generator_function():
    assert inspect.isgeneratorfunction(evidence_repository.get_patient_evidence)
    assert inspect.isgeneratorfunction(evidence_repository.get_patient_evidence_by_code)


def test_evidence_service_does_not_eagerly_materialize_generator_results():
    import app.services.evidence_service as module

    source = inspect.getsource(module)
    # The service is a thin pass-through: it must not wrap repository
    # generators in list() itself (callers decide whether/when to consume).
    assert "list(" not in source


# --- 3: patient-scoped queries never become unscoped ----------------------------------------


def test_evidence_repository_queries_are_always_patient_scoped(mongo_db, monkeypatch):
    _insert_observations(mongo_db, "p1", 3)
    captured = []
    original_find = mongo_db["fhir_resources"].find

    def spy_find(query, *args, **kwargs):
        captured.append(dict(query))
        return original_find(query, *args, **kwargs)

    monkeypatch.setattr(mongo_db["fhir_resources"], "find", spy_find)

    list(evidence_repository.get_patient_evidence(mongo_db, "p1"))
    evidence_repository.list_patient_evidence(mongo_db, "p1", page=1, page_size=10)

    assert captured  # at least one query was issued
    for query in captured:
        assert query.get("patient_id") == "p1"
        assert query != {}


def test_audit_repository_queries_are_always_patient_scoped(mongo_db, monkeypatch):
    audit_repository.insert_audit_record(
        mongo_db,
        AuditRecord(
            audit_id="a1",
            patient_id="p1",
            query="q",
            retrieval_status="evidence_found",
            answer_status="answered",
            evidence_references=[],
            created_at=datetime.now(timezone.utc),
        ),
    )
    captured = []
    original_find = mongo_db["audit_records"].find

    def spy_find(query, *args, **kwargs):
        captured.append(dict(query))
        return original_find(query, *args, **kwargs)

    monkeypatch.setattr(mongo_db["audit_records"], "find", spy_find)

    audit_repository.get_patient_audit_history(mongo_db, "p1")

    assert captured == [{"patient_id": "p1"}]


# --- 4/5: pagination limits returned records (repository + evidence API) -------------------


def test_list_patient_evidence_pagination_bounds_results(mongo_db):
    _insert_observations(mongo_db, "p1", 30)

    items, total = evidence_repository.list_patient_evidence(mongo_db, "p1", page=1, page_size=10)

    assert total == 30
    assert len(items) == 10  # never the full 30-item history


def test_evidence_api_does_not_retrieve_unlimited_records(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _insert_observations(mongo_db, "profile-patient-1", 50)

    resp = api_client.get("/api/patients/profile-patient-1/evidence?resource_type=Observation")

    body = resp.json()
    assert body["pagination"]["total"] >= 51  # 50 bulk + 1 fixture observation
    assert len(body["items"]) == body["pagination"]["page_size"]
    assert len(body["items"]) < body["pagination"]["total"]


# --- 6: audit API does not retrieve unlimited records ---------------------------------------


def test_audit_api_does_not_retrieve_unlimited_records(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider("noted [Condition/cond-1]."))
    for _ in range(15):
        api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hypertension"})
    _clear_llm_override()

    resp = api_client.get("/api/patients/profile-patient-1/audit?page=1&page_size=5")

    body = resp.json()
    assert body["pagination"]["total"] == 15
    assert len(body["items"]) == 5  # never the full 15-record history


# --- 7: cross-patient isolation remains intact -----------------------------------------------


def test_cross_patient_isolation_still_holds_for_evidence_and_audit(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    _override_llm(_FakeLLMProvider("noted [Condition/cond-2]."))
    api_client.post("/api/patients/profile-patient-2/ask", json={"query": "diabetes"})
    _clear_llm_override()

    evidence_resp = api_client.get("/api/patients/profile-patient-1/evidence")
    audit_resp_1 = api_client.get("/api/patients/profile-patient-1/audit")
    audit_resp_2 = api_client.get("/api/patients/profile-patient-2/audit")

    assert all(item["patient_id"] == "profile-patient-1" for item in evidence_resp.json()["items"])
    assert audit_resp_1.json()["pagination"]["total"] == 0
    assert audit_resp_2.json()["pagination"]["total"] == 1


# --- 8/9: existing indexes remain present, no duplicates ------------------------------------


_EXPECTED_INDEXES = {
    "fhir_resources": {"_id_", "uniq_resource_type_id", "idx_patient_id", "idx_patient_resource_type"},
    "patients": {"_id_", "uniq_patient_id"},
    "patient_profiles": {"_id_", "uniq_profile_patient_id"},
    "clinical_trials": {"_id_", "uniq_trial_id", "idx_trial_status", "idx_trial_conditions"},
    "audit_records": {"_id_", "idx_patient_created_at"},
}


def test_expected_indexes_are_present(mongo_db):
    for collection_name, expected_names in _EXPECTED_INDEXES.items():
        actual_names = {ix["name"] for ix in mongo_db[collection_name].list_indexes()}
        assert expected_names.issubset(actual_names), f"{collection_name} missing indexes: {expected_names - actual_names}"


def test_ensure_indexes_is_idempotent_and_creates_no_duplicates(mongo_db):
    ensure_indexes(mongo_db)
    ensure_indexes(mongo_db)  # calling twice must not create duplicate/renamed indexes

    for collection_name, expected_names in _EXPECTED_INDEXES.items():
        actual_names = {ix["name"] for ix in mongo_db[collection_name].list_indexes()}
        assert actual_names == expected_names, f"{collection_name}: {actual_names} != {expected_names}"


# --- 10: ask pipeline still limits grounded evidence -----------------------------------------


def test_hybrid_retriever_default_limit_is_unchanged():
    assert HYBRID_DEFAULT_LIMIT == 5


def test_ask_pipeline_never_exceeds_the_default_evidence_limit(mongo_db):
    for i in range(20):
        mongo_db["fhir_resources"].insert_one(
            {
                "patient_id": "p1",
                "resource_type": "Observation",
                "resource_id": f"obs-{i}",
                "data": {
                    "resourceType": "Observation",
                    "id": f"obs-{i}",
                    "status": "final",
                    "code": {"coding": [{"system": "http://loinc.org", "code": "8480-6", "display": "Systolic Blood Pressure"}]},
                    "effectiveDateTime": "2023-01-10",
                    "valueQuantity": {"value": 120 + i, "unit": "mm[Hg]"},
                },
            }
        )

    result = HybridEvidenceRetriever(mongo_db).retrieve(RetrievalRequest(patient_id="p1", query="blood pressure"))

    assert len(result.evidence) <= HYBRID_DEFAULT_LIMIT


# --- 11/12: unsupported / no-evidence queries still avoid the LLM ---------------------------


def test_unsupported_query_still_avoids_llm_call(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    fake = _FakeLLMProvider()
    _override_llm(fake)

    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "???"})

    _clear_llm_override()
    assert resp.json()["status"] == "unsupported"
    assert fake.calls == []


def test_no_evidence_query_still_avoids_llm_call(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)
    fake = _FakeLLMProvider()
    _override_llm(fake)

    resp = api_client.post("/api/patients/profile-patient-1/ask", json={"query": "hemoglobin a1c"})

    _clear_llm_override()
    assert resp.json()["status"] == "insufficient_evidence"
    assert fake.calls == []


# --- 13: existing eligibility semantics remain unchanged -------------------------------------


def test_eligibility_pass_fail_unknown_precedence_unchanged():
    profile = PatientProfileOut(
        patient_id="p1",
        demographics=DemographicsOut(patient_id="p1", date_of_birth="1980-01-01", gender="female"),
        contact=ContactInfoOut(),
        observations=[],
    )
    trial = ClinicalTrialOut(
        trial_id="T-SCALE-CHECK",
        status="recruiting",
        eligibility=TrialEligibilityOut(
            minimum_age=18,
            inclusion_criteria=[
                EligibilityCriterionOut(type="observation_threshold", label="HbA1c", code="4548-4", operator="<", value=9.0)
            ],
        ),
    )

    result = EligibilityMatcher().evaluate(profile, trial)

    # Age passes (known), HbA1c is UNKNOWN (missing) -> overall UNKNOWN, not
    # promoted to ELIGIBLE and not demoted to INELIGIBLE.
    assert result.eligibility_status == "UNKNOWN"
    assert any(c.result == "PASS" for c in result.matched_criteria)
    assert any(c.result == "UNKNOWN" for c in result.unknown_criteria)
    assert result.failed_criteria == []


# --- 14: existing API behavior remains compatible ---------------------------------------------


def test_existing_endpoints_remain_functional_and_compatible(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    assert api_client.get("/health").status_code == 200
    assert api_client.get("/api/patients").status_code == 200

    resources_resp = api_client.get("/api/patients/profile-patient-1/resources")
    assert resources_resp.status_code == 200
    assert "pagination" in resources_resp.json()

    evidence_resp = api_client.get("/api/patients/profile-patient-1/evidence")
    assert evidence_resp.status_code == 200
    assert "pagination" in evidence_resp.json()

    audit_resp = api_client.get("/api/patients/profile-patient-1/audit")
    assert audit_resp.status_code == 200
    assert "pagination" in audit_resp.json()
    assert audit_resp.json()["items"] == []

    matches_resp = api_client.get("/api/patients/profile-patient-1/matches")
    assert matches_resp.status_code == 200
    assert "total_trials_evaluated" in matches_resp.json()
