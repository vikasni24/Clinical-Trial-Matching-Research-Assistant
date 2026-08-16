import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getPatientAuditHistory } from "../api/audit";
import { useScopedQuery } from "../hooks/useScopedQuery";
import { SkeletonRows } from "../components/common/LoadingState";
import { ErrorBlock } from "../components/common/ErrorState";
import { EmptyBlock } from "../components/common/EmptyState";
import { PaginationBar } from "../components/common/Pagination";
import { AnswerStatusBadge, RetrievalStatusBadge } from "../components/common/StatusBadge";
import { AuditOutcomeChart, type AuditOutcomeDatum } from "../components/charts/AuditOutcomeChart";
import { formatDateTime } from "../utils/format";
import type { AnswerStatus } from "../types/api";

const PAGE_SIZE = 10;
const RECENT_SUMMARY_SIZE = 50;

const STATUS_LABEL: Record<AnswerStatus, string> = {
  answered: "Answered",
  insufficient_evidence: "Insufficient evidence",
  unsupported: "Unsupported",
};

function useRecentOutcomeBreakdown(patientId: string | undefined) {
  return useScopedQuery(patientId ? `audit-outcomes-${patientId}` : null, async () => {
    const result = await getPatientAuditHistory(patientId!, 1, RECENT_SUMMARY_SIZE);
    const tally: Record<AnswerStatus, number> = { answered: 0, insufficient_evidence: 0, unsupported: 0 };
    for (const record of result.items) tally[record.answer_status]++;
    const data: AuditOutcomeDatum[] = (["answered", "insufficient_evidence", "unsupported"] as const)
      .map((status) => ({ status, label: STATUS_LABEL[status], count: tally[status] }))
      .filter((d) => d.count > 0);
    return { data, sampleSize: result.items.length, total: result.pagination.total };
  });
}

export function AuditHistory() {
  const { patientId } = useParams<{ patientId: string }>();
  const [page, setPage] = useState(1);

  const audit = useScopedQuery(patientId ? `audit-${patientId}-${page}` : null, () =>
    getPatientAuditHistory(patientId!, page, PAGE_SIZE)
  );
  const breakdown = useRecentOutcomeBreakdown(patientId);

  if (!patientId) return null;

  return (
    <div>
      <div className="breadcrumb">
        <Link to="/patients">Patients</Link> / <Link to={`/patients/${patientId}`}>{patientId}</Link> / Audit history
      </div>
      <div className="app-topbar">
        <div className="page-header-text">
          <h1>Audit history</h1>
          <p>A traceable record of every grounded question asked about this patient.</p>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 16, alignItems: "start" }}>
        <div className="card">
          {audit.loading && (
            <div className="card-padded">
              <SkeletonRows rows={5} />
            </div>
          )}

          {audit.error && (
            <div className="card-padded">
              <ErrorBlock error={audit.error} onRetry={() => setPage((p) => p)} />
            </div>
          )}

          {audit.data && audit.data.items.length === 0 && (
            <EmptyBlock title="No audit history yet" hint="This patient hasn't been asked any research assistant questions yet." />
          )}

          {audit.data && audit.data.items.length > 0 && (
            <>
              <div>
                {audit.data.items.map((record) => (
                  <div key={record.audit_id} style={{ padding: "14px 20px", borderTop: "1px solid var(--border)", position: "relative" }}>
                    <div
                      style={{
                        position: "absolute",
                        left: 0,
                        top: 14,
                        bottom: 14,
                        width: 3,
                        borderRadius: 2,
                        background:
                          record.answer_status === "answered"
                            ? "var(--green)"
                            : record.answer_status === "insufficient_evidence"
                              ? "var(--amber)"
                              : "var(--gray)",
                      }}
                    />
                    <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 8, paddingLeft: 10 }}>
                      <div style={{ fontWeight: 600, fontSize: 13.5 }}>{record.query}</div>
                      <span className="text-faint" style={{ fontSize: 12 }}>{formatDateTime(record.created_at)}</span>
                    </div>

                    <div className="pill-row" style={{ marginTop: 8, paddingLeft: 10 }}>
                      <RetrievalStatusBadge status={record.retrieval_status} />
                      <AnswerStatusBadge status={record.answer_status} />
                    </div>

                    {record.evidence_references.length > 0 && (
                      <div style={{ marginTop: 8, fontSize: 12, paddingLeft: 10 }}>
                        <span className="text-faint">Evidence referenced: </span>
                        {record.evidence_references.map((ref, i) => (
                          <span key={`${ref.resource_type}/${ref.resource_id}`}>
                            <span className="text-mono">{ref.resource_type}/{ref.resource_id}</span>
                            {i < record.evidence_references.length - 1 ? ", " : ""}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
              <div className="card-padded" style={{ paddingTop: 12 }}>
                <PaginationBar pagination={audit.data.pagination} onPageChange={setPage} />
              </div>
            </>
          )}
        </div>

        <div className="card card-padded">
          <h3>Recent outcomes</h3>
          <p className="text-faint" style={{ fontSize: 11.5, marginTop: 4 }}>
            {breakdown.data ? `Last ${breakdown.data.sampleSize} of ${breakdown.data.total} questions` : " "}
          </p>
          <div style={{ marginTop: 10 }}>
            {breakdown.loading && <SkeletonRows rows={3} />}
            {breakdown.error && <ErrorBlock error={breakdown.error} />}
            {breakdown.data && breakdown.data.data.length === 0 && (
              <p className="text-muted" style={{ fontSize: 13 }}>No questions asked yet.</p>
            )}
            {breakdown.data && breakdown.data.data.length > 0 && <AuditOutcomeChart data={breakdown.data.data} />}
          </div>
        </div>
      </div>
    </div>
  );
}
