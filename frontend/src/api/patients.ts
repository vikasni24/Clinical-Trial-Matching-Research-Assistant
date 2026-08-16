import { getJson } from "./client";
import type { PatientListOut, PatientOut, PatientProfileOut } from "../types/api";

export function listPatients(page: number, pageSize: number): Promise<PatientListOut> {
  return getJson<PatientListOut>("/api/patients", { page, page_size: pageSize });
}

export function getPatient(patientId: string): Promise<PatientOut> {
  return getJson<PatientOut>(`/api/patients/${encodeURIComponent(patientId)}`);
}

export function getPatientProfile(patientId: string): Promise<PatientProfileOut> {
  return getJson<PatientProfileOut>(`/api/patients/${encodeURIComponent(patientId)}/profile`);
}
