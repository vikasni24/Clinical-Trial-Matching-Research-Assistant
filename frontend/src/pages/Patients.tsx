import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { listPatients } from "../api/patients";
import { getPatientEvidence } from "../api/evidence";
import { useScopedQuery } from "../hooks/useScopedQuery";
import { SkeletonRows } from "../components/common/LoadingState";
import { ErrorBlock } from "../components/common/ErrorState";
import { EmptyBlock } from "../components/common/EmptyState";
import { PaginationBar } from "../components/common/Pagination";
import { summarizePatientResource } from "../utils/fhir";
import { formatDate } from "../utils/format";

const PAGE_SIZE = 20;

/** Fetches a per-patient evidence total ONLY for the patient IDs on the
 * currently displayed page — bounded by page size, never every patient in
 * the system. Each count comes directly from
 * GET /api/patients/{id}/evidence's own pagination.total. */
function useEvidenceCounts(patientIds: string[]) {
  const [counts, setCounts] = useState<Record<string, number | null>>({});

  useEffect(() => {
    let cancelled = false;
    setCounts({});
    Promise.all(
      patientIds.map(async (id) => {
        try {
          const result = await getPatientEvidence(id, 1, 1);
          return [id, result.pagination.total] as const;
        } catch {
          return [id, null] as const;
        }
      })
    ).then((entries) => {
      if (cancelled) return;
      setCounts(Object.fromEntries(entries));
    });
    return () => {
      cancelled = true;
    };
  }, [patientIds.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps

  return counts;
}

export function Patients() {
  const navigate = useNavigate();
  const [page, setPage] = useState(1);

  const query = useScopedQuery(`patients-page-${page}`, () => listPatients(page, PAGE_SIZE));
  const evidenceCounts = useEvidenceCounts(query.data?.items.map((p) => p.patient_id) ?? []);

  return (
    <div>
      <div className="app-topbar">
        <div className="page-header-text">
          <h1>Patients</h1>
          <p>All patients currently stored from ingested FHIR/Synthea data.</p>
        </div>
      </div>

      <div className="card">
        {query.loading && (
          <div className="card-padded">
            <SkeletonRows rows={6} height={22} />
          </div>
        )}

        {query.error && (
          <div className="card-padded">
            <ErrorBlock error={query.error} onRetry={() => setPage((p) => p)} />
          </div>
        )}

        {query.data && query.data.items.length === 0 && (
          <EmptyBlock title="No patients found" hint="No patient data has been ingested into the backend yet." />
        )}

        {query.data && query.data.items.length > 0 && (
          <>
            <table>
              <thead>
                <tr>
                  <th>Patient</th>
                  <th>Gender</th>
                  <th>Date of birth</th>
                  <th>Evidence</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {query.data.items.map((patient) => {
                  const summary = summarizePatientResource(patient.data);
                  const count = evidenceCounts[patient.patient_id];
                  return (
                    <tr key={patient.patient_id}>
                      <td data-label="Patient">
                        <div style={{ fontWeight: 600 }}>{summary.name ?? <span className="text-faint">Unknown name</span>}</div>
                        <div className="text-mono" style={{ marginTop: 2 }}>{patient.patient_id}</div>
                      </td>
                      <td data-label="Gender">{summary.gender ?? <span className="text-faint">—</span>}</td>
                      <td data-label="Date of birth">{formatDate(summary.birthDate)}</td>
                      <td data-label="Evidence">
                        {count === undefined && <span className="skeleton" style={{ display: "inline-block", width: 28, height: 14 }} />}
                        {count === null && <span className="text-faint">—</span>}
                        {typeof count === "number" && <span className="badge badge-primary">{count}</span>}
                      </td>
                      <td data-label="Actions">
                        <div className="pill-row">
                          <button type="button" className="btn btn-ghost" onClick={() => navigate(`/patients/${patient.patient_id}`)}>
                            View
                          </button>
                          <button
                            type="button"
                            className="btn btn-ghost"
                            onClick={() => navigate(`/patients/${patient.patient_id}/matching`)}
                          >
                            Match trials
                          </button>
                          <button
                            type="button"
                            className="btn btn-ghost"
                            onClick={() => navigate(`/patients/${patient.patient_id}/assistant`)}
                          >
                            Ask assistant
                          </button>
                          <button
                            type="button"
                            className="btn btn-ghost"
                            onClick={() => navigate(`/patients/${patient.patient_id}/audit`)}
                          >
                            Audit history
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div className="card-padded" style={{ paddingTop: 0 }}>
              <PaginationBar pagination={query.data.pagination} onPageChange={setPage} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
