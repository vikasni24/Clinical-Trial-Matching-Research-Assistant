/**
 * Reads a small, well-known set of fields off a raw FHIR Patient resource
 * (PatientOut.data) for compact display purposes only — e.g. the patient
 * list. This never renders the raw FHIR document itself; it extracts a
 * handful of named fields into a plain display string, the same way the
 * backend's own patient_normalization.py does server-side.
 */
export interface PatientDisplaySummary {
  name: string | null;
  gender: string | null;
  birthDate: string | null;
}

interface FhirHumanName {
  given?: string[];
  family?: string;
}

export function summarizePatientResource(data: Record<string, unknown>): PatientDisplaySummary {
  const names = Array.isArray(data.name) ? (data.name as FhirHumanName[]) : [];
  const primary = names[0];
  const given = primary?.given?.join(" ") ?? "";
  const family = primary?.family ?? "";
  const fullName = [given, family].filter(Boolean).join(" ").trim();

  return {
    name: fullName || null,
    gender: typeof data.gender === "string" ? data.gender : null,
    birthDate: typeof data.birthDate === "string" ? data.birthDate : null,
  };
}
