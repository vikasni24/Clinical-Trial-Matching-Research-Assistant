import { getJson } from "./client";
import type { AuditHistoryOut } from "../types/api";

export function getPatientAuditHistory(patientId: string, page: number, pageSize: number): Promise<AuditHistoryOut> {
  return getJson<AuditHistoryOut>(`/api/patients/${encodeURIComponent(patientId)}/audit`, {
    page,
    page_size: pageSize,
  });
}
