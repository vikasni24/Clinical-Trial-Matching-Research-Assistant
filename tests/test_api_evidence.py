"""Phase 4E: proves evidence is correctly, safely exposed through the API —
both via the existing GET /matches/{trial_id} (which gained `evidence` on
each criterion in Phase 4D with zero route code changes) and the new
GET /api/patients/{id}/evidence endpoint for general-purpose browsing.

Existing Phase 1-4D API tests (test_api_patients.py, test_api_fhir.py,
test_api_patient_profile.py, test_api_matches.py, test_api_trials.py) are
run as part of the full suite and are not modified here."""

import shutil

from app.services.fhir_ingestion import FHIRIngestionService
from app.services.patient_normalization import PatientNormalizationService
from app.services.trial_ingestion import TrialIngestionService


def _setup(mongo_db, tmp_path, fixtures_dir):
    shutil.copy(fixtures_dir / "profile_bundle.json", tmp_path / "profile_bundle.json")
    FHIRIngestionService(mongo_db, fhir_dir=tmp_path).run()
    PatientNormalizationService(mongo_db).normalize_all()

    trials_dir = tmp_path / "trials"
    trials_dir.mkdir()
    shutil.copy(fixtures_dir / "trials_matching_fixture.json", trials_dir / "trials_matching_fixture.json")
    TrialIngestionService(mongo_db, trials_path=trials_dir / "trials_matching_fixture.json").run()


_EVIDENCE_ALLOWED_KEYS = {
    "patient_id",
    "resource_type",
    "resource_id",
    "source_collection",
    "source_reference",
    "code",
    "coding_system",
    "display",
    "value",
    "unit",
    "effective_date",
    "status",
}


# --- match endpoint exposes structured evidence (1-11) -----------------------


def test_match_endpoint_returns_structured_evidence(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/matches/MATCH-TRIAL-ELIGIBLE")

    assert resp.status_code == 200
    body = resp.json()
    condition_eval = next(c for c in body["matched_criteria"] if c["category"] == "condition")
    assert isinstance(condition_eval["evidence"], list)
    assert len(condition_eval["evidence"]) == 1


def test_evidence_contains_resource_id(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/matches/MATCH-TRIAL-ELIGIBLE")
    condition_eval = next(c for c in resp.json()["matched_criteria"] if c["category"] == "condition")

    assert condition_eval["evidence"][0]["resource_id"] == "cond-1"


def test_evidence_contains_source_reference(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/matches/MATCH-TRIAL-ELIGIBLE")
    condition_eval = next(c for c in resp.json()["matched_criteria"] if c["category"] == "condition")

    assert condition_eval["evidence"][0]["source_reference"] == "Condition/cond-1"


def test_evidence_contains_patient_id(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/matches/MATCH-TRIAL-ELIGIBLE")
    condition_eval = next(c for c in resp.json()["matched_criteria"] if c["category"] == "condition")

    assert condition_eval["evidence"][0]["patient_id"] == "profile-patient-1"


def test_observation_evidence_correctly_exposed(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/evidence?resource_type=Observation")

    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["resource_type"] == "Observation"
    assert items[0]["value"] == 125
    assert items[0]["unit"] == "mm[Hg]"


def test_condition_evidence_correctly_exposed(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/evidence?resource_type=Condition")

    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["display"] == "Hypertension"
    assert items[0]["code"] == "38341003"


def test_medication_evidence_correctly_exposed(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/evidence?resource_type=MedicationRequest")

    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) == 1
    assert items[0]["code"] == "860975"


def test_allergy_evidence_correctly_exposed(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/matches/MATCH-TRIAL-INELIGIBLE")

    assert resp.status_code == 200
    body = resp.json()
    allergy_eval = next(c for c in body["failed_criteria"] if c["category"] == "allergy")
    assert allergy_eval["evidence"][0]["resource_type"] == "AllergyIntolerance"
    assert allergy_eval["evidence"][0]["resource_id"] == "allergy-1"


def test_missing_evidence_remains_empty_list(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/matches/MATCH-TRIAL-UNKNOWN")

    body = resp.json()
    hba1c_eval = body["unknown_criteria"][0]
    assert hba1c_eval["evidence"] == []


def test_unknown_criterion_is_not_converted_to_fail(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/matches/MATCH-TRIAL-UNKNOWN")

    body = resp.json()
    assert body["eligibility_status"] == "UNKNOWN"
    assert body["failed_criteria"] == []
    assert len(body["unknown_criteria"]) == 1


def test_unknown_criterion_is_not_converted_to_pass(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/matches/MATCH-TRIAL-UNKNOWN")

    body = resp.json()
    matched_categories = [c["category"] for c in body["matched_criteria"]]
    assert "observation" not in matched_categories


# --- not-found / isolation / determinism (12-13, 16-18) -----------------------


def test_unknown_patient_returns_404_on_match_endpoint(api_client):
    resp = api_client.get("/api/patients/does-not-exist/matches/MATCH-TRIAL-ELIGIBLE")

    assert resp.status_code == 404


def test_patient_isolation_at_api_level(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp1 = api_client.get("/api/patients/profile-patient-1/matches/MATCH-TRIAL-DIABETES")
    resp2 = api_client.get("/api/patients/profile-patient-2/matches/MATCH-TRIAL-DIABETES")

    assert resp1.json()["eligibility_status"] == "INELIGIBLE"
    assert resp2.json()["eligibility_status"] == "ELIGIBLE"
    condition_eval_2 = next(c for c in resp2.json()["matched_criteria"] if c["category"] == "condition")
    assert condition_eval_2["evidence"][0]["patient_id"] == "profile-patient-2"
    assert condition_eval_2["evidence"][0]["resource_id"] == "cond-2"


def test_existing_phase_1_to_3_apis_remain_functional(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    assert api_client.get("/health").status_code == 200
    assert api_client.get("/api/patients").status_code == 200
    assert api_client.get("/api/patients/profile-patient-1").status_code == 200
    assert api_client.get("/api/patients/profile-patient-1/resources").status_code == 200
    assert api_client.get("/api/patients/profile-patient-1/profile").status_code == 200
    assert api_client.get("/api/fhir/Condition/cond-1").status_code == 200
    assert api_client.get("/api/trials").status_code == 200


def test_existing_match_api_behavior_remains_backward_compatible(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/matches")
    body = resp.json()

    assert resp.status_code == 200
    assert "patient_id" in body
    assert "total_trials_evaluated" in body
    assert "matches" in body
    for match in body["matches"]:
        assert {"patient_id", "trial_id", "eligibility_status", "match_score", "matched_criteria", "explanation"}.issubset(
            match.keys()
        )


def test_repeated_identical_requests_produce_identical_evidence(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    first = api_client.get("/api/patients/profile-patient-1/matches/MATCH-TRIAL-ELIGIBLE").json()
    second = api_client.get("/api/patients/profile-patient-1/matches/MATCH-TRIAL-ELIGIBLE").json()

    first.pop("evaluated_at", None)
    second.pop("evaluated_at", None)
    assert first == second


# --- data minimization (14-15) --------------------------------------------------


def test_no_raw_fhir_document_in_evidence_response(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/evidence?resource_type=Condition")

    for item in resp.json()["items"]:
        assert "data" not in item
        assert "resourceType" not in item
        assert set(item.keys()).issubset(_EVIDENCE_ALLOWED_KEYS)


def test_no_mongo_id_in_evidence_response(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    match_resp = api_client.get("/api/patients/profile-patient-1/matches/MATCH-TRIAL-ELIGIBLE")
    for c in match_resp.json()["matched_criteria"]:
        for evidence in c["evidence"]:
            assert "_id" not in evidence

    evidence_resp = api_client.get("/api/patients/profile-patient-1/evidence")
    for item in evidence_resp.json()["items"]:
        assert "_id" not in item


# --- dedicated /evidence endpoint (19-24) ---------------------------------------


def test_evidence_endpoint_is_patient_scoped(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/evidence")

    assert resp.status_code == 200
    items = resp.json()["items"]
    assert len(items) > 0
    assert all(item["patient_id"] == "profile-patient-1" for item in items)


def test_evidence_endpoint_resource_type_filtering(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/evidence?resource_type=Procedure")

    body = resp.json()
    assert body["pagination"]["total"] == 1
    assert body["items"][0]["resource_type"] == "Procedure"
    assert body["items"][0]["resource_id"] == "proc-1"


def test_evidence_endpoint_code_filtering(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/evidence?code=38341003")

    body = resp.json()
    assert body["pagination"]["total"] == 1
    assert body["items"][0]["resource_type"] == "Condition"
    assert body["items"][0]["display"] == "Hypertension"


def test_evidence_endpoint_code_filtering_no_match_returns_empty(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/evidence?code=does-not-exist")

    body = resp.json()
    assert body["items"] == []
    assert body["pagination"]["total"] == 0


def test_evidence_endpoint_pagination(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp = api_client.get("/api/patients/profile-patient-1/evidence?page=1&page_size=1")

    body = resp.json()
    assert len(body["items"]) == 1
    assert body["pagination"]["page_size"] == 1
    assert body["pagination"]["total"] > 1
    assert body["pagination"]["total_pages"] > 1


def test_evidence_endpoint_unknown_patient_returns_404(api_client):
    resp = api_client.get("/api/patients/does-not-exist/evidence")

    assert resp.status_code == 404


def test_evidence_endpoint_cross_patient_isolation(api_client, mongo_db, tmp_path, fixtures_dir):
    _setup(mongo_db, tmp_path, fixtures_dir)

    resp1 = api_client.get("/api/patients/profile-patient-1/evidence")
    resp2 = api_client.get("/api/patients/profile-patient-2/evidence")

    ids_1 = {item["resource_id"] for item in resp1.json()["items"]}
    ids_2 = {item["resource_id"] for item in resp2.json()["items"]}
    assert ids_1.isdisjoint(ids_2)
    assert "cond-2" not in ids_1  # patient-2's Diabetes condition must never appear for patient-1
    assert "cond-1" not in ids_2  # patient-1's Hypertension condition must never appear for patient-2
