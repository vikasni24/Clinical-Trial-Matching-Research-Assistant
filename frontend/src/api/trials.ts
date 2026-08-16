import { getJson } from "./client";
import type { ClinicalTrialOut, TrialListOut } from "../types/api";

export function listTrials(page: number, pageSize: number, status?: string): Promise<TrialListOut> {
  return getJson<TrialListOut>("/api/trials", { page, page_size: pageSize, status });
}

export function getTrial(trialId: string): Promise<ClinicalTrialOut> {
  return getJson<ClinicalTrialOut>(`/api/trials/${encodeURIComponent(trialId)}`);
}
