import { useState, type ReactNode } from "react";
import { NavLink, useParams } from "react-router-dom";
import { getHealth } from "../../api/system";
import { listPatients } from "../../api/patients";
import { useScopedQuery } from "../../hooks/useScopedQuery";
import {
  IconAssistant,
  IconAudit,
  IconClose,
  IconDashboard,
  IconMenu,
  IconPatients,
  IconTrials,
} from "../common/Icons";

const navLinkClass = ({ isActive }: { isActive: boolean }) => `app-nav-link${isActive ? " active" : ""}`;

function BrandMark() {
  return (
    <div className="app-brand">
      <div className="app-brand-mark">
        <svg viewBox="0 0 24 24" fill="none" stroke="#04121a" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 3v6M12 15v6M3 12h6M15 12h6" />
          <circle cx="12" cy="12" r="2.3" fill="#04121a" stroke="none" />
        </svg>
      </div>
      <div className="app-brand-text">
        <span className="app-brand-title">Clinical Trial Research AI</span>
        <span className="app-brand-subtitle">EVIDENCE-GROUNDED</span>
      </div>
    </div>
  );
}

function StatusIndicator({ label, state }: { label: string; state: "online" | "offline" | "pending" }) {
  const stateLabel = state === "online" ? "Online" : state === "offline" ? "Offline" : "Checking…";
  return (
    <div className="status-row">
      <span className="text-muted">{label}</span>
      <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span className={`status-dot ${state}`} />
        <span className="text-faint">{stateLabel}</span>
      </span>
    </div>
  );
}

function SidebarStatus() {
  const health = useScopedQuery("shell-health", getHealth);
  const data = useScopedQuery("shell-data", () => listPatients(1, 1));

  const backendState = health.loading ? "pending" : health.error ? "offline" : "online";
  const dataState = data.loading ? "pending" : data.error ? "offline" : "online";

  return (
    <div className="app-sidebar-footer">
      <StatusIndicator label="Backend API" state={backendState} />
      <StatusIndicator label="Patient Data" state={dataState} />
    </div>
  );
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  const { patientId } = useParams();

  return (
    <>
      <BrandMark />

      <nav className="app-nav">
        <NavLink to="/" end className={navLinkClass} onClick={onNavigate}>
          <IconDashboard /> Dashboard
        </NavLink>
        <NavLink to="/patients" className={navLinkClass} onClick={onNavigate}>
          <IconPatients /> Patients
        </NavLink>
        <NavLink to="/trials" className={navLinkClass} onClick={onNavigate}>
          <IconTrials /> Clinical Trials
        </NavLink>
      </nav>

      {patientId && (
        <nav className="app-nav">
          <div className="app-nav-section-label">Current patient</div>
          <NavLink to={`/patients/${patientId}`} end className={navLinkClass} onClick={onNavigate}>
            <IconPatients /> Overview
          </NavLink>
          <NavLink to={`/patients/${patientId}/matching`} className={navLinkClass} onClick={onNavigate}>
            <IconTrials /> Trial Matching
          </NavLink>
          <NavLink to={`/patients/${patientId}/assistant`} className={navLinkClass} onClick={onNavigate}>
            <IconAssistant /> Research Assistant
          </NavLink>
          <NavLink to={`/patients/${patientId}/audit`} className={navLinkClass} onClick={onNavigate}>
            <IconAudit /> Audit
          </NavLink>
        </nav>
      )}

      <SidebarStatus />
    </>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="app-shell">
      <aside className={`app-sidebar${mobileOpen ? " open" : ""}`}>
        <SidebarContent onNavigate={() => setMobileOpen(false)} />
      </aside>

      {mobileOpen && (
        <div
          onClick={() => setMobileOpen(false)}
          style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", zIndex: 15 }}
        />
      )}

      <main className="app-main">
        <div className="mobile-topbar">
          <button type="button" className="btn btn-ghost" onClick={() => setMobileOpen(true)} aria-label="Open menu">
            <IconMenu />
          </button>
          {mobileOpen && (
            <button type="button" className="btn btn-ghost" onClick={() => setMobileOpen(false)} aria-label="Close menu">
              <IconClose />
            </button>
          )}
        </div>
        {children}
      </main>
    </div>
  );
}
