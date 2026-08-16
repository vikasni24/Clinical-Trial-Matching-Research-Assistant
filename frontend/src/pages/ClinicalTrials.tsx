import { useState } from "react";
import { listTrials } from "../api/trials";
import { useScopedQuery } from "../hooks/useScopedQuery";
import { SkeletonRows } from "../components/common/LoadingState";
import { ErrorBlock } from "../components/common/ErrorState";
import { EmptyBlock } from "../components/common/EmptyState";
import { PaginationBar } from "../components/common/Pagination";

const PAGE_SIZE = 10;
const STATUS_OPTIONS = ["", "recruiting", "completed", "closed", "active"];

export function ClinicalTrials() {
  const [page, setPage] = useState(1);
  const [status, setStatus] = useState("");

  const trials = useScopedQuery(`trials-browse-${page}-${status}`, () => listTrials(page, PAGE_SIZE, status || undefined));

  return (
    <div>
      <div className="app-topbar">
        <div className="page-header-text">
          <h1>Clinical Trials</h1>
          <p>Browse the synthetic trial catalog ingested from the local trials fixture.</p>
        </div>
      </div>

      {trials.data?.disclaimer && <div className="disclaimer-banner">{trials.data.disclaimer}</div>}

      <div className="card">
        <div className="card-padded" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
          <h2>All trials</h2>
          <select
            className="input"
            style={{ width: 200 }}
            value={status}
            onChange={(e) => {
              setStatus(e.target.value);
              setPage(1);
            }}
          >
            {STATUS_OPTIONS.map((s) => (
              <option key={s} value={s}>
                {s ? s[0].toUpperCase() + s.slice(1) : "All statuses"}
              </option>
            ))}
          </select>
        </div>

        {trials.loading && (
          <div className="card-padded" style={{ paddingTop: 0 }}>
            <SkeletonRows rows={6} />
          </div>
        )}

        {trials.error && (
          <div className="card-padded" style={{ paddingTop: 0 }}>
            <ErrorBlock error={trials.error} onRetry={() => setPage((p) => p)} />
          </div>
        )}

        {trials.data && trials.data.items.length === 0 && (
          <div className="card-padded" style={{ paddingTop: 0 }}>
            <EmptyBlock title="No trials found" hint="No trials match the selected status filter." />
          </div>
        )}

        {trials.data && trials.data.items.length > 0 && (
          <>
            <table>
              <thead>
                <tr>
                  <th>Trial</th>
                  <th>Status</th>
                  <th>Phase</th>
                  <th>Conditions</th>
                  <th>Sponsor</th>
                </tr>
              </thead>
              <tbody>
                {trials.data.items.map((trial) => (
                  <tr key={trial.trial_id}>
                    <td data-label="Trial">
                      <div style={{ fontWeight: 600 }}>{trial.brief_title ?? trial.trial_id}</div>
                      <div className="text-mono" style={{ marginTop: 2 }}>{trial.trial_id}</div>
                    </td>
                    <td data-label="Status">
                      <span className="badge badge-primary">{trial.status ?? "unknown"}</span>
                    </td>
                    <td data-label="Phase">{trial.phase ?? <span className="text-faint">—</span>}</td>
                    <td data-label="Conditions">
                      <div className="pill-row">
                        {trial.conditions.length === 0 && <span className="text-faint">—</span>}
                        {trial.conditions.slice(0, 3).map((c) => (
                          <span key={c.name} className="badge badge-neutral">{c.name}</span>
                        ))}
                      </div>
                    </td>
                    <td data-label="Sponsor" className="text-muted">{trial.sponsor ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="card-padded" style={{ paddingTop: 0 }}>
              <PaginationBar pagination={trials.data.pagination} onPageChange={setPage} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
