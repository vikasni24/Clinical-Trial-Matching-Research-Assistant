import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { listPatients } from "../api/patients";
import { listTrials } from "../api/trials";
import { getHealth } from "../api/system";
import { useScopedQuery } from "../hooks/useScopedQuery";
import { SkeletonRows } from "../components/common/LoadingState";
import { ErrorBlock } from "../components/common/ErrorState";
import {
  IconAssistant,
  IconAudit,
  IconChevronRight,
  IconDashboard,
  IconPatients,
  IconPulse,
  IconSearch,
  IconTrials,
} from "../components/common/Icons";
import { TrialStatusChart, type TrialStatusDatum } from "../components/charts/TrialStatusChart";
import { TrialPhaseChart, type TrialPhaseDatum } from "../components/charts/TrialPhaseChart";

const TRIAL_STATUSES = ["recruiting", "active", "completed", "closed"];

// Only 17 trials exist in the real dataset — one bounded fetch of the whole
// catalog, tallied client-side, since GET /api/trials has no phase filter.
const TRIAL_CATALOG_PAGE_SIZE = 200;

function useTrialStatusBreakdown() {
  return useScopedQuery("dashboard-trial-status-breakdown", async () => {
    const results = await Promise.all(TRIAL_STATUSES.map((status) => listTrials(1, 1, status)));
    const data: TrialStatusDatum[] = TRIAL_STATUSES.map((status, i) => ({
      status: status[0].toUpperCase() + status.slice(1),
      count: results[i].pagination.total,
    })).filter((d) => d.count > 0);
    return data;
  });
}

function useTrialPhaseBreakdown() {
  return useScopedQuery("dashboard-trial-phase-breakdown", async () => {
    const catalog = await listTrials(1, TRIAL_CATALOG_PAGE_SIZE);
    const counts = new Map<string, number>();
    for (const trial of catalog.items) {
      const label = trial.phase?.trim() || "Unspecified";
      counts.set(label, (counts.get(label) ?? 0) + 1);
    }
    const data: TrialPhaseDatum[] = Array.from(counts.entries())
      .map(([phase, count]) => ({ phase, count }))
      .sort((a, b) => b.count - a.count);
    return data;
  });
}

export function Dashboard() {
  const navigate = useNavigate();
  const [patientIdInput, setPatientIdInput] = useState("");

  const patientTotals = useScopedQuery("dashboard-patients", () => listPatients(1, 1));
  const trialTotals = useScopedQuery("dashboard-trials", () => listTrials(1, 1));
  const health = useScopedQuery("dashboard-health", getHealth);
  const trialBreakdown = useTrialStatusBreakdown();
  const phaseBreakdown = useTrialPhaseBreakdown();

  function goToPatient(destination: "" | "matching" | "assistant" | "audit") {
    const id = patientIdInput.trim();
    if (!id) return;
    const suffix = destination ? `/${destination}` : "";
    navigate(`/patients/${encodeURIComponent(id)}${suffix}`);
  }

  return (
    <div>
      <div className="hero-panel">
        <div className="hero-eyebrow">Clinical Intelligence Platform</div>
        <h1 className="hero-title">Command Center</h1>
        <p className="hero-subtitle">Patient evidence → trial matching → grounded research insights, every claim traceable back to source data.</p>
        <div className="hero-pipeline">
          <span className="hero-pipeline-step"><IconPulse /> FHIR evidence</span>
          <span className="hero-pipeline-arrow"><IconChevronRight /></span>
          <span className="hero-pipeline-step"><IconTrials /> Trial matching</span>
          <span className="hero-pipeline-arrow"><IconChevronRight /></span>
          <span className="hero-pipeline-step"><IconAssistant /> Research assistant</span>
          <span className="hero-pipeline-arrow"><IconChevronRight /></span>
          <span className="hero-pipeline-step"><IconAudit /> Audit trail</span>
        </div>
      </div>

      <div className="stat-grid">
        <KpiCard
          icon={<IconPatients />}
          accent="blue"
          label="Total Patients"
          value={patientTotals.data ? String(patientTotals.data.pagination.total) : null}
          loading={patientTotals.loading}
          error={!!patientTotals.error}
          hint="GET /api/patients"
        />
        <KpiCard
          icon={<IconTrials />}
          accent="green"
          label="Active Clinical Trials"
          value={trialTotals.data ? String(trialTotals.data.pagination.total) : null}
          loading={trialTotals.loading}
          error={!!trialTotals.error}
          hint="GET /api/trials"
        />
        <KpiCard
          icon={<IconPulse />}
          accent="green"
          label="Backend"
          value={health.data ? "ONLINE" : health.error ? "OFFLINE" : null}
          loading={health.loading}
          error={!!health.error}
          hint="GET /health"
          valueTone={health.error ? "danger" : "success"}
        />
        <KpiCard
          icon={<IconAssistant />}
          accent={health.data?.llm_configured ? "green" : "gray"}
          label="AI Assistant"
          value={
            health.loading
              ? null
              : health.error
                ? null
                : health.data?.llm_configured
                  ? `READY (${health.data.llm_provider ?? "configured"})`
                  : "NOT CONFIGURED"
          }
          loading={health.loading}
          error={!!health.error}
          hint={health.data?.llm_configured ? "GET /health · llm_configured" : "No LLM_API_KEY configured on the backend"}
          valueTone={health.data?.llm_configured ? "success" : "neutral"}
        />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, alignItems: "start", marginBottom: 16 }}>
        <div className="card card-padded">
          <h3>Trial status distribution</h3>
          <p className="text-faint" style={{ fontSize: 12, marginTop: 4 }}>
            Live counts per recruitment status, from the trial catalog.
          </p>
          <div style={{ marginTop: 10 }}>
            {trialBreakdown.loading && <SkeletonRows rows={4} height={26} />}
            {trialBreakdown.error && <ErrorBlock error={trialBreakdown.error} />}
            {trialBreakdown.data && trialBreakdown.data.length === 0 && (
              <p className="text-muted" style={{ fontSize: 13 }}>No trials ingested yet.</p>
            )}
            {trialBreakdown.data && trialBreakdown.data.length > 0 && <TrialStatusChart data={trialBreakdown.data} />}
          </div>
        </div>

        <div className="card card-padded">
          <h3>Trial phase distribution</h3>
          <p className="text-faint" style={{ fontSize: 12, marginTop: 4 }}>
            Live counts per study phase, across the full trial catalog.
          </p>
          <div style={{ marginTop: 10 }}>
            {phaseBreakdown.loading && <SkeletonRows rows={4} height={26} />}
            {phaseBreakdown.error && <ErrorBlock error={phaseBreakdown.error} />}
            {phaseBreakdown.data && phaseBreakdown.data.length === 0 && (
              <p className="text-muted" style={{ fontSize: 13 }}>No trials ingested yet.</p>
            )}
            {phaseBreakdown.data && phaseBreakdown.data.length > 0 && <TrialPhaseChart data={phaseBreakdown.data} />}
          </div>
        </div>
      </div>

      <div className="card card-padded" style={{ marginBottom: 20 }}>
        <h3>Quick patient lookup</h3>
        <p className="text-faint" style={{ fontSize: 12, marginTop: 4, marginBottom: 14 }}>
          Jump directly to a patient's record, trial matching, research assistant, or audit history.
        </p>
        <div style={{ display: "flex", gap: 8, maxWidth: 480 }}>
          <div style={{ position: "relative", flex: 1 }}>
            <span
              style={{
                position: "absolute",
                left: 12,
                top: "50%",
                transform: "translateY(-50%)",
                color: "var(--text-faint)",
                display: "flex",
                pointerEvents: "none",
              }}
            >
              <IconSearch />
            </span>
            <input
              className="input"
              style={{ paddingLeft: 34, width: "100%" }}
              placeholder="Patient ID"
              value={patientIdInput}
              onChange={(e) => setPatientIdInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") goToPatient("");
              }}
            />
          </div>
          <button type="button" className="btn btn-primary" onClick={() => goToPatient("")}>
            Open
          </button>
        </div>
        <div className="pill-row" style={{ marginTop: 12 }}>
          <button type="button" className="btn btn-secondary" onClick={() => goToPatient("matching")}>
            <IconTrials /> Trial matching
          </button>
          <button type="button" className="btn btn-secondary" onClick={() => goToPatient("assistant")}>
            <IconAssistant /> Research assistant
          </button>
          <button type="button" className="btn btn-secondary" onClick={() => goToPatient("audit")}>
            <IconAudit /> Audit history
          </button>
          <button type="button" className="btn btn-ghost" onClick={() => navigate("/patients")}>
            <IconDashboard /> Browse all patients
          </button>
        </div>
      </div>

      {(patientTotals.error || trialTotals.error) && <ErrorBlock error={(patientTotals.error ?? trialTotals.error)!} />}
    </div>
  );
}

function KpiCard({
  icon,
  accent,
  label,
  value,
  loading,
  error,
  hint,
  valueTone,
}: {
  icon: React.ReactNode;
  accent: "blue" | "green" | "gray";
  label: string;
  value: string | null;
  loading: boolean;
  error: boolean;
  hint: string;
  valueTone?: "success" | "danger" | "neutral";
}) {
  const accentVar = accent === "blue" ? "var(--blue-glow)" : accent === "green" ? "var(--green-glow)" : "transparent";
  const iconBg = accent === "blue" ? "var(--blue-surface)" : accent === "green" ? "var(--green-surface)" : "var(--gray-surface)";
  const iconColor = accent === "blue" ? "var(--blue-bright)" : accent === "green" ? "var(--green)" : "var(--gray)";
  const valueColor =
    valueTone === "success" ? "var(--green)" : valueTone === "danger" ? "var(--red)" : valueTone === "neutral" ? "var(--gray)" : "var(--text)";

  return (
    <div className="stat-card" style={{ ["--accent-glow" as string]: accentVar }}>
      <div className="stat-card-icon" style={{ background: iconBg, color: iconColor }}>
        {icon}
      </div>
      <div className="stat-card-label">{label}</div>
      {loading && <SkeletonRows rows={1} height={26} />}
      {!loading && error && !value && <div className="stat-card-value" style={{ color: "var(--red)", fontSize: 16 }}>Unavailable</div>}
      {!loading && value && (
        <div className="stat-card-value" style={{ color: valueColor, fontSize: valueTone ? 18 : 28 }}>
          {value}
        </div>
      )}
      <div className="stat-card-hint">{hint}</div>
    </div>
  );
}
