import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { listTrials } from "../api/trials";
import { getPatientTrialMatch, listPatientMatches } from "../api/matches";
import { useScopedQuery } from "../hooks/useScopedQuery";
import { LoadingBlock, SkeletonRows } from "../components/common/LoadingState";
import { ErrorBlock } from "../components/common/ErrorState";
import { EmptyBlock } from "../components/common/EmptyState";
import { OverallEligibilityBadge } from "../components/common/StatusBadge";
import { CriterionRow } from "../components/matching/CriterionRow";
import { PaginationBar } from "../components/common/Pagination";
import { EligibilityDonut, type EligibilityDonutDatum } from "../components/charts/EligibilityDonut";
import type { CriterionEvaluationOut, OverallEligibilityStatus } from "../types/api";

const TRIALS_PAGE_SIZE = 10;

function useEligibilityBreakdown(patientId: string | undefined) {
  return useScopedQuery(patientId ? `eligibility-breakdown-${patientId}` : null, async () => {
    const result = await listPatientMatches(patientId!, "all");
    const tally: Record<OverallEligibilityStatus, number> = { ELIGIBLE: 0, INELIGIBLE: 0, UNKNOWN: 0 };
    for (const match of result.matches) tally[match.eligibility_status]++;
    const data: EligibilityDonutDatum[] = (["ELIGIBLE", "INELIGIBLE", "UNKNOWN"] as const)
      .map((status) => ({ status, count: tally[status] }))
      .filter((d) => d.count > 0);
    return { data, total: result.total_trials_evaluated };
  });
}

export function TrialMatching() {
  const { patientId } = useParams<{ patientId: string }>();
  const [trialsPage, setTrialsPage] = useState(1);
  const [selectedTrialId, setSelectedTrialId] = useState<string | null>(null);

  const trials = useScopedQuery(`trials-page-${trialsPage}`, () => listTrials(trialsPage, TRIALS_PAGE_SIZE));
  const breakdown = useEligibilityBreakdown(patientId);

  const match = useScopedQuery(
    patientId && selectedTrialId ? `match-${patientId}-${selectedTrialId}` : null,
    () => getPatientTrialMatch(patientId!, selectedTrialId!)
  );

  if (!patientId) return null;

  return (
    <div>
      <div className="breadcrumb">
        <Link to="/patients">Patients</Link> / <Link to={`/patients/${patientId}`}>{patientId}</Link> / Trial matching
      </div>
      <div className="app-topbar">
        <div className="page-header-text">
          <h1>Trial matching</h1>
          <p>Deterministic PASS / FAIL / UNKNOWN eligibility evaluation.</p>
        </div>
      </div>

      {trials.data?.disclaimer && <div className="disclaimer-banner">{trials.data.disclaimer}</div>}

      <div style={{ display: "grid", gridTemplateColumns: "260px 1.4fr 1fr", gap: 16, alignItems: "start" }}>
        <div className="card">
          <div className="card-padded" style={{ paddingBottom: 8 }}>
            <h3>Trial discovery</h3>
          </div>
          {trials.loading && (
            <div className="card-padded">
              <SkeletonRows rows={5} />
            </div>
          )}
          {trials.error && (
            <div className="card-padded">
              <ErrorBlock error={trials.error} />
            </div>
          )}
          {trials.data && trials.data.items.length === 0 && (
            <div className="card-padded">
              <EmptyBlock title="No trials available" hint="No clinical trials have been ingested yet." />
            </div>
          )}
          {trials.data && trials.data.items.length > 0 && (
            <>
              <div>
                {trials.data.items.map((trial) => (
                  <button
                    key={trial.trial_id}
                    type="button"
                    onClick={() => setSelectedTrialId(trial.trial_id)}
                    className="glow-border"
                    style={{
                      display: "block",
                      width: "100%",
                      textAlign: "left",
                      padding: "10px 16px",
                      border: "none",
                      borderTop: "1px solid var(--border)",
                      background: trial.trial_id === selectedTrialId ? "var(--blue-surface)" : "transparent",
                      cursor: "pointer",
                      transition: "background 160ms ease",
                    }}
                  >
                    <div style={{ fontWeight: 600, fontSize: 13 }}>{trial.brief_title ?? trial.trial_id}</div>
                    <div className="text-faint" style={{ fontSize: 11.5, marginTop: 2 }}>
                      {trial.trial_id} · {trial.status ?? "unknown status"}
                    </div>
                  </button>
                ))}
              </div>
              <div className="card-padded" style={{ paddingTop: 8 }}>
                <PaginationBar pagination={trials.data.pagination} onPageChange={setTrialsPage} />
              </div>
            </>
          )}
        </div>

        <div className="card card-padded">
          {!selectedTrialId && (
            <EmptyBlock title="Select a trial" hint="Choose a trial from the discovery panel to run eligibility analysis." />
          )}

          {selectedTrialId && match.loading && <LoadingBlock label="Evaluating eligibility…" />}
          {selectedTrialId && match.error && <ErrorBlock error={match.error} />}

          {selectedTrialId && match.data && (
            <div>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
                <div>
                  <h3 style={{ marginBottom: 4 }}>Eligibility analysis</h3>
                  <h2>{match.data.trial_id}</h2>
                  <p className="text-muted" style={{ marginTop: 6 }}>{match.data.explanation}</p>
                </div>
                <OverallEligibilityBadge status={match.data.eligibility_status} />
              </div>

              <div className="pill-row" style={{ marginTop: 10, fontSize: 12.5 }}>
                <span className="text-faint">Match score: {(match.data.match_score * 100).toFixed(0)}%</span>
                {match.data.overall_status && <span className="text-faint">Recruitment status: {match.data.overall_status}</span>}
              </div>

              <CriteriaSection title="Failed criteria" criteria={match.data.failed_criteria} />
              <CriteriaSection title="Unknown criteria" criteria={match.data.unknown_criteria} />
              <CriteriaSection title="Matched criteria" criteria={match.data.matched_criteria} />
            </div>
          )}
        </div>

        <div className="card card-padded">
          <h3>Eligibility distribution</h3>
          <p className="text-faint" style={{ fontSize: 11.5, marginTop: 4 }}>Across all evaluated trials for this patient.</p>
          <div style={{ marginTop: 8 }}>
            {breakdown.loading && <SkeletonRows rows={3} />}
            {breakdown.error && <ErrorBlock error={breakdown.error} />}
            {breakdown.data && breakdown.data.total === 0 && (
              <p className="text-muted" style={{ fontSize: 13 }}>No candidate trials evaluated.</p>
            )}
            {breakdown.data && breakdown.data.total > 0 && (
              <>
                <EligibilityDonut data={breakdown.data.data} />
                <div style={{ display: "flex", flexDirection: "column", gap: 6, marginTop: 8 }}>
                  {breakdown.data.data.map((d) => (
                    <div key={d.status} style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5 }}>
                      <OverallEligibilityBadge status={d.status} />
                      <span className="text-muted">{d.count}</span>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function CriteriaSection({ title, criteria }: { title: string; criteria: CriterionEvaluationOut[] }) {
  if (criteria.length === 0) return null;
  return (
    <div style={{ marginTop: 20 }}>
      <h3>{title}</h3>
      <div style={{ marginTop: 8 }}>
        {criteria.map((c, i) => (
          <CriterionRow key={`${c.criterion}-${i}`} criterion={c} />
        ))}
      </div>
    </div>
  );
}
