import { getJson } from "./client";
import type { EvidenceListOut } from "../types/api";

export function getPatientEvidence(
  patientId: string,
  page: number,
  pageSize: number,
  resourceType?: string
): Promise<EvidenceListOut> {
  return getJson<EvidenceListOut>(`/api/patients/${encodeURIComponent(patientId)}/evidence`, {
    page,
    page_size: pageSize,
    resource_type: resourceType,
  });
}
