import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { Dashboard } from "./pages/Dashboard";
import { Patients } from "./pages/Patients";
import { PatientDetail } from "./pages/PatientDetail";
import { ClinicalTrials } from "./pages/ClinicalTrials";
import { TrialMatching } from "./pages/TrialMatching";
import { ResearchAssistant } from "./pages/ResearchAssistant";
import { AuditHistory } from "./pages/AuditHistory";

export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/patients" element={<Patients />} />
        <Route path="/patients/:patientId" element={<PatientDetail />} />
        <Route path="/trials" element={<ClinicalTrials />} />
        <Route path="/patients/:patientId/matching" element={<TrialMatching />} />
        <Route path="/patients/:patientId/assistant" element={<ResearchAssistant />} />
        <Route path="/patients/:patientId/audit" element={<AuditHistory />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  );
}
