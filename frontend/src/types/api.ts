/**
 * TypeScript types mirroring the backend's actual Pydantic response models
 * exactly (see app/models/*.py). Every field here was verified against the
 * real backend source — nothing here is guessed or invented.
 *
 * Three-state semantics are preserved verbatim as string literal unions,
 * never collapsed or renamed:
 *   - Eligibility: "PASS" | "FAIL" | "UNKNOWN"          (app/models/match_result.py)
 *   - Retrieval:   "evidence_found" | "no_evidence_found" | "unsupported" (app/models/retrieval.py)
 *   - Answer:      "answered" | "insufficient_evidence" | "unsupported"  (app/models/answer.py)
 */

// --- shared -----------------------------------------------------------------

export interface PaginationMeta {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
}

// --- patients (app/models/schemas.py) ----------------------------------------

export interface PatientOut {
  patient_id: string;
  source_file: string | null;
  ingested_at: string | null;
  /** The original, verbatim FHIR Patient resource. Only ever read for
   * specific known fields (name/gender/birthDate) client-side — never
   * rendered as raw JSON in the UI. See src/utils/fhir.ts. */
  data: Record<string, unknown>;
}

export interface PatientListOut {
  items: PatientOut[];
  pagination: PaginationMeta;
}

export interface FHIRResourceOut {
  resource_type: string;
  resource_id: string;
  patient_id: string | null;
  source_file: string | null;
  ingested_at: string | null;
  data: Record<string, unknown>;
}

export interface ResourceListOut {
  items: FHIRResourceOut[];
  pagination: PaginationMeta;
}

// --- patient profile (app/models/patient_profile.py) --------------------------

export interface DemographicsOut {
  patient_id: string;
  first_name: string | null;
  last_name: string | null;
  full_name: string | null;
  date_of_birth: string | null;
  gender: string | null;
  race: string | null;
  ethnicity: string | null;
  marital_status: string | null;
  deceased: boolean;
  deceased_date: string | null;
}

export interface TelecomOut {
  system: string | null;
  value: string | null;
  use: string | null;
}

export interface ContactInfoOut {
  address: string | null;
  city: string | null;
  state: string | null;
  postal_code: string | null;
  country: string | null;
  telecom: TelecomOut[];
}

export interface PatientProfileOut {
  patient_id: string;
  demographics: DemographicsOut;
  contact: ContactInfoOut;
  conditions: unknown[];
  observations: unknown[];
  medications: unknown[];
  procedures: unknown[];
  encounters: unknown[];
  diagnostic_reports: unknown[];
  allergies: unknown[];
  normalized_at: string | null;
}

// --- evidence (app/models/evidence.py) -----------------------------------------

export interface Evidence {
  patient_id: string;
  resource_type: string;
  resource_id: string;
  source_collection: string;
  source_reference: string | null;
  code: string | null;
  coding_system: string | null;
  display: string | null;
  value: unknown;
  unit: string | null;
  effective_date: string | null;
  status: string | null;
}

export interface EvidenceListOut {
  items: Evidence[];
  pagination: PaginationMeta;
}

// --- clinical trials (app/models/clinical_trial.py) -----------------------------

export interface TrialConditionOut {
  name: string;
  code: string | null;
  system: string | null;
}

export interface TrialInterventionOut {
  name: string;
  type: string | null;
  description: string | null;
}

export interface TrialLocationOut {
  facility: string | null;
  city: string | null;
  state: string | null;
  country: string | null;
  recruiting_status: string | null;
}

export interface EligibilityCriterionOut {
  type: string;
  label: string;
  code: string | null;
  system: string | null;
  display: string | null;
  operator: string | null;
  value: number | null;
  unit: string | null;
}

export interface TrialEligibilityOut {
  minimum_age: number | null;
  maximum_age: number | null;
  sex: string | null;
  inclusion_criteria: EligibilityCriterionOut[];
  exclusion_criteria: EligibilityCriterionOut[];
}

export interface ClinicalTrialOut {
  trial_id: string;
  title: string | null;
  brief_title: string | null;
  official_title: string | null;
  sponsor: string | null;
  study_type: string | null;
  phase: string | null;
  status: string | null;
  conditions: TrialConditionOut[];
  interventions: TrialInterventionOut[];
  eligibility: TrialEligibilityOut;
  locations: TrialLocationOut[];
  source: string | null;
  source_id: string | null;
  source_url: string | null;
  last_updated: string | null;
  ingested_at: string | null;
}

export interface TrialListOut {
  items: ClinicalTrialOut[];
  pagination: PaginationMeta;
  disclaimer: string;
}

// --- eligibility / matching (app/models/match_result.py) -----------------------

export type EligibilityResult = "PASS" | "FAIL" | "UNKNOWN";
export type OverallEligibilityStatus = "ELIGIBLE" | "INELIGIBLE" | "UNKNOWN";

export interface CriterionEvaluationOut {
  criterion: string;
  category: string;
  requirement: string;
  result: EligibilityResult;
  patient_value: unknown;
  required_value: string | null;
  reason: string;
  evidence: Evidence[];
}

export interface MatchResultOut {
  patient_id: string;
  trial_id: string;
  overall_status: string | null;
  eligibility_status: OverallEligibilityStatus;
  match_score: number;
  matched_criteria: CriterionEvaluationOut[];
  failed_criteria: CriterionEvaluationOut[];
  unknown_criteria: CriterionEvaluationOut[];
  explanation: string;
  evaluated_at: string | null;
}

export interface MatchListOut {
  patient_id: string;
  total_trials_evaluated: number;
  matches: MatchResultOut[];
}

// --- research assistant (app/models/answer.py) ----------------------------------

export type AnswerStatus = "answered" | "insufficient_evidence" | "unsupported";

export interface AskRequest {
  query: string;
}

export interface GroundedAnswer {
  patient_id: string;
  query: string;
  status: AnswerStatus;
  answer_text: string | null;
  evidence: Evidence[];
  message: string | null;
  /** Distinguishes "no evidence was ever retrieved" from "evidence existed
   * but the generated answer wasn't grounded in it" — both otherwise
   * collapse into status="insufficient_evidence". May be null for older
   * backend versions that don't send it. */
  retrieval_status: RetrievalStatus | null;
}

// --- audit (app/models/audit.py) --------------------------------------------------

export type RetrievalStatus = "evidence_found" | "no_evidence_found" | "unsupported";

export interface AuditEvidenceReferenceOut {
  resource_type: string;
  resource_id: string;
}

export interface AuditRecordOut {
  audit_id: string;
  patient_id: string;
  query: string;
  retrieval_status: RetrievalStatus;
  answer_status: AnswerStatus;
  evidence_references: AuditEvidenceReferenceOut[];
  created_at: string;
}

export interface AuditHistoryOut {
  items: AuditRecordOut[];
  pagination: PaginationMeta;
}
