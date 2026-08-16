import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { listPatients } from "../api/patients";
import { listTrials } from "../api/trials";
import { getHealth } from "../api/system";
import { useScopedQuery } from "../hooks/useScopedQuery";
import { SkeletonRows } from "../components/common/LoadingState";
import { ErrorBlock } from "../components/common/ErrorState";
import { IconAssistant, IconDashboard, IconPatients, IconPulse, IconTrials } from "../components/common/Icons";
import { TrialStatusChart, type TrialStatusDatum } from "../components/charts/TrialStatusChart";

const TRIAL_STATUSES = ["recruiting", "active", "completed", "closed"];

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

export function Dashboard() {
  const navigate = useNavigate();
  const [patientIdInput, setPatientIdInput] = useState("");

  const patientTotals = useScopedQuery("dashboard-patients", () => listPatients(1, 1));
  const trialTotals = useScopedQuery("dashboard-trials", () => listTrials(1, 1));
  const health = useScopedQuery("dashboard-health", getHealth);
  const trialBreakdown = useTrialStatusBreakdown();

  function goToPatient(destination: "" | "matching" | "assistant" | "audit") {
    const id = patientIdInput.trim();
    if (!id) return;
    const suffix = destination ? `/${destination}` : "";
    navigate(`/patients/${encodeURIComponent(id)}${suffix}`);
  }

  return (
    <div>
      <div className="app-topbar">
        <div className="page-header-text">
          <h1>Command Center</h1>
          <p>Real-time overview of the clinical trial matching and research assistant system.</p>
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
          accent="gray"
          label="AI Assistant"
          value="NOT VERIFIED"
          loading={false}
          error={false}
          hint="Configuration status isn't exposed by any endpoint"
          valueTone="neutral"
        />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1.3fr 1fr", gap: 16, alignItems: "start", marginBottom: 20 }}>
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
          <h3>Quick patient lookup</h3>
          <p className="text-faint" style={{ fontSize: 12, marginTop: 4, marginBottom: 14 }}>
            Jump directly to a patient's record, trial matching, research assistant, or audit history.
          </p>
          <div style={{ display: "flex", gap: 8 }}>
            <input
              className="input"
              placeholder="Patient ID"
              value={patientIdInput}
              onChange={(e) => setPatientIdInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") goToPatient("");
              }}
            />
            <button type="button" className="btn btn-primary" onClick={() => goToPatient("")}>
              Open
            </button>
          </div>
          <div className="pill-row" style={{ marginTop: 12 }}>
            <button type="button" className="btn btn-secondary" onClick={() => goToPatient("matching")}>
              Trial matching
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => goToPatient("assistant")}>
              Research assistant
            </button>
            <button type="button" className="btn btn-secondary" onClick={() => goToPatient("audit")}>
              Audit history
            </button>
          </div>
          <div className="divider" />
          <button type="button" className="btn btn-ghost" style={{ width: "100%" }} onClick={() => navigate("/patients")}>
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
