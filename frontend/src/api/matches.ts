import { getJson } from "./client";
import type { MatchListOut, MatchResultOut } from "../types/api";

export function listPatientMatches(patientId: string, status?: string): Promise<MatchListOut> {
  return getJson<MatchListOut>(`/api/patients/${encodeURIComponent(patientId)}/matches`, { status });
}

export function getPatientTrialMatch(patientId: string, trialId: string): Promise<MatchResultOut> {
  return getJson<MatchResultOut>(
    `/api/patients/${encodeURIComponent(patientId)}/matches/${encodeURIComponent(trialId)}`
  );
}
