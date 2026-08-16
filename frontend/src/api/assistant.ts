import { postJson } from "./client";
import type { GroundedAnswer } from "../types/api";

export function askPatientQuestion(patientId: string, query: string): Promise<GroundedAnswer> {
  return postJson<GroundedAnswer>(`/api/patients/${encodeURIComponent(patientId)}/ask`, { query });
}
