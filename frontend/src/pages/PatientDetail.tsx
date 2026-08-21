import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { getPatientProfile } from "../api/patients";
import { getPatientEvidence } from "../api/evidence";
import { useScopedQuery, type ScopedQueryState } from "../hooks/useScopedQuery";
import { LoadingBlock, SkeletonRows } from "../components/common/LoadingState";
import { ErrorBlock } from "../components/common/ErrorState";
import { EmptyBlock } from "../components/common/EmptyState";
import { PaginationBar } from "../components/common/Pagination";
import { EvidenceCardList } from "../components/evidence/EvidenceCard";
import { EvidenceCategoryChart, type EvidenceCategoryDatum } from "../components/charts/EvidenceCategoryChart";
import { ObservationTrendChart } from "../components/charts/ObservationTrendChart";
import { IconAssistant, IconAudit, IconTrials } from "../components/common/Icons";
import { formatDate, formatDateTime } from "../utils/format";
import { groupObservationTrends } from "../utils/observationGrouping";
import type { EvidenceListOut } from "../types/api";

const PAGE_SIZE = 12;
const RESOURCE_TYPES = [
  "", "Condition", "Observation", "MedicationRequest", "Procedure", "Encounter", "DiagnosticReport", "AllergyIntolerance",
];

function useEvidenceCategoryTotals(patientId: string | undefined) {
  return useScopedQuery(patientId ? `evidence-categories-${patientId}` : null, async () => {
    const types = RESOURCE_TYPES.filter(Boolean);
    const results = await Promise.all(types.map((t) => getPatientEvidence(patientId!, 1, 1, t)));
    const data: EvidenceCategoryDatum[] = types
      .map((t, i) => ({ resourceType: t, count: results[i].pagination.total }))
      .filter((d) => d.count > 0)
      .sort((a, b) => b.count - a.count);
    return data;
  });
}

export function PatientDetail() {
  const { patientId } = useParams<{ patientId: string }>();
  const navigate = useNavigate();
  const [page, setPage] = useState(1);
  const [resourceType, setResourceType] = useState("");

  const profile = useScopedQuery(patientId ? `profile-${patientId}` : null, () => getPatientProfile(patientId!));
  const categoryTotals = useEvidenceCategoryTotals(patientId);
  const observations = useScopedQuery(patientId ? `obs-trend-${patientId}` : null, () =>
    getPatientEvidence(patientId!, 1, 100, "Observation")
  );
  const conditions = useScopedQuery(patientId ? `conditions-${patientId}` : null, () =>
    getPatientEvidence(patientId!, 1, 20, "Condition")
  );
  const medications = useScopedQuery(patientId ? `medications-${patientId}` : null, () =>
    getPatientEvidence(patientId!, 1, 20, "MedicationRequest")
  );
  const allergies = useScopedQuery(patientId ? `allergies-${patientId}` : null, () =>
    getPatientEvidence(patientId!, 1, 20, "AllergyIntolerance")
  );
  const browse = useScopedQuery(
    patientId ? `evidence-browse-${patientId}-${page}-${resourceType}` : null,
    () => getPatientEvidence(patientId!, page, PAGE_SIZE, resourceType || undefined)
  );

  if (!patientId) return null;

  const totalEvidence = categoryTotals.data?.reduce((sum, d) => sum + d.count, 0) ?? null;
  const trends = observations.data ? groupObservationTrends(observations.data.items) : [];

  return (
    <div>
      <div className="breadcrumb">
        <Link to="/patients">Patients</Link> / {patientId}
      </div>

      <div className="app-topbar">
        <div className="page-header-text">
          <h1>{profile.data?.demographics.full_name ?? "Patient"}</h1>
          <p className="text-mono" style={{ marginTop: 4 }}>{patientId}</p>
        </div>
        <div className="pill-row">
          <button type="button" className="btn btn-secondary" onClick={() => navigate(`/patients/${patientId}/matching`)}>
            <IconTrials /> Trial matching
          </button>
          <button type="button" className="btn btn-primary" onClick={() => navigate(`/patients/${patientId}/assistant`)}>
            <IconAssistant /> Research assistant
          </button>
          <button type="button" className="btn btn-secondary" onClick={() => navigate(`/patients/${patientId}/audit`)}>
            <IconAudit /> Audit history
          </button>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.3fr 1fr", gap: 16, alignItems: "start", marginBottom: 16 }}>
        <div className="card card-padded">
          <h3>Clinical overview</h3>
          {profile.loading && <SkeletonRows rows={3} />}
          {profile.error && <ErrorBlock error={profile.error} onRetry={() => setPage((p) => p)} />}
          {profile.data && (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 14, marginTop: 14 }}>
              <Field label="Gender" value={profile.data.demographics.gender} />
              <Field label="Date of birth" value={formatDate(profile.data.demographics.date_of_birth)} />
              <Field label="Race" value={profile.data.demographics.race} />
              <Field label="Ethnicity" value={profile.data.demographics.ethnicity} />
              <Field label="Marital status" value={profile.data.demographics.marital_status} />
              <Field label="Location" value={[profile.data.contact.city, profile.data.contact.state].filter(Boolean).join(", ") || null} />
              <Field label="Profile normalized" value={formatDateTime(profile.data.normalized_at)} />
            </div>
          )}
        </div>

        <div className="card card-padded">
          <h3>Evidence summary</h3>
          {categoryTotals.loading && <SkeletonRows rows={4} height={20} />}
          {categoryTotals.error && <ErrorBlock error={categoryTotals.error} />}
          {categoryTotals.data && categoryTotals.data.length === 0 && (
            <p className="text-muted" style={{ fontSize: 13, marginTop: 10 }}>No evidence recorded yet.</p>
          )}
          {categoryTotals.data && categoryTotals.data.length > 0 && (
            <>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginTop: 4, marginBottom: 6 }}>
                <span style={{ fontSize: 26, fontWeight: 700 }}>{totalEvidence}</span>
                <span className="text-faint" style={{ fontSize: 12 }}>traceable evidence items</span>
              </div>
              <EvidenceCategoryChart data={categoryTotals.data} />
            </>
          )}
        </div>
      </div>

      {trends.length > 0 && (
        <div className="card card-padded" style={{ marginBottom: 16 }}>
          <h3>Key observations</h3>
          <p className="text-faint" style={{ fontSize: 12, marginTop: 4, marginBottom: 10 }}>
            Historical trends for recorded numeric observations (up to the 100 most recently indexed).
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 14 }}>
            {trends.slice(0, 4).map((trend) => (
              <div key={trend.label} style={{ border: "1px solid var(--border)", borderRadius: "var(--radius-md)", padding: "12px 14px" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
                  <span style={{ fontWeight: 600, fontSize: 13 }}>{trend.label}</span>
                  <span className="text-faint" style={{ fontSize: 11.5 }}>{trend.points.length} readings</span>
                </div>
                <ObservationTrendChart data={trend.points} unit={trend.unit} />
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16, marginBottom: 16 }}>
        <CategoryPanel title="Conditions" query={conditions} emptyHint="No conditions recorded." />
        <CategoryPanel title="Medications" query={medications} emptyHint="No medications recorded." />
        <CategoryPanel title="Allergies" query={allergies} emptyHint="No allergies recorded." />
      </div>

      <div className="card card-padded">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
          <h3>Evidence browser</h3>
          <select
            className="input"
            style={{ width: 200 }}
            value={resourceType}
            onChange={(e) => {
              setResourceType(e.target.value);
              setPage(1);
            }}
          >
            {RESOURCE_TYPES.map((type) => (
              <option key={type} value={type}>
                {type || "All resource types"}
              </option>
            ))}
          </select>
        </div>

        <div style={{ marginTop: 14 }}>
          {browse.loading && <LoadingBlock label="Loading evidence…" />}
          {browse.error && <ErrorBlock error={browse.error} onRetry={() => setPage((p) => p)} />}
          {browse.data && browse.data.items.length === 0 && (
            <EmptyBlock
              title="No evidence found"
              hint={resourceType ? `This patient has no recorded ${resourceType} evidence.` : "This patient has no recorded evidence yet."}
            />
          )}
          {browse.data && browse.data.items.length > 0 && (
            <>
              <EvidenceCardList evidence={browse.data.items} />
              <PaginationBar pagination={browse.data.pagination} onPageChange={setPage} />
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function CategoryPanel({
  title,
  query,
  emptyHint,
}: {
  title: string;
  query: ScopedQueryState<EvidenceListOut>;
  emptyHint: string;
}) {
  return (
    <div className="card card-padded">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <h3>{title}</h3>
        {query.data && <span className="text-faint" style={{ fontSize: 11.5 }}>{query.data.pagination.total}</span>}
      </div>
      <div style={{ marginTop: 10 }}>
        {query.loading && <SkeletonRows rows={2} height={18} />}
        {query.error && <ErrorBlock error={query.error} />}
        {query.data && query.data.items.length === 0 && <p className="text-faint" style={{ fontSize: 12.5 }}>{emptyHint}</p>}
        {query.data && query.data.items.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {query.data.items.map((item) => (
              <div key={`${item.resource_type}/${item.resource_id}`} style={{ fontSize: 12.5 }}>
                <div style={{ fontWeight: 600 }}>{item.display ?? "Unnamed"}</div>
                <div className="text-faint">
                  {item.status && <span>{item.status} · </span>}
                  {formatDate(item.effective_date)}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string | null | undefined }) {
  return (
    <div>
      <div style={{ fontSize: 10.5, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
        {label}
      </div>
      <div style={{ marginTop: 3, fontSize: 13.5 }}>{value || <span className="text-faint">Not recorded</span>}</div>
    </div>
  );
}
