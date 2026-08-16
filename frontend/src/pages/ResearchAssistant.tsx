import { useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { askPatientQuestion } from "../api/assistant";
import { getPatientEvidence } from "../api/evidence";
import { ApiError } from "../api/client";
import { useScopedQuery } from "../hooks/useScopedQuery";
import { AnswerStatusBadge } from "../components/common/StatusBadge";
import { ErrorBlock } from "../components/common/ErrorState";
import { EvidenceCardList } from "../components/evidence/EvidenceCard";
import { IconAssistant } from "../components/common/Icons";
import type { GroundedAnswer } from "../types/api";

const SUGGESTED_QUESTIONS = [
  "What is the patient's latest HbA1c?",
  "What medications is the patient taking?",
  "What relevant conditions are documented?",
  "What evidence is available for hypertension?",
  "Summarize the patient's relevant clinical evidence.",
];

const LOADING_PHASES = ["Retrieving patient evidence…", "Grounding response…", "Generating answer…"];

function useLoadingPhase(active: boolean) {
  const [phaseIndex, setPhaseIndex] = useState(0);
  useEffect(() => {
    if (!active) {
      setPhaseIndex(0);
      return;
    }
    const interval = setInterval(() => setPhaseIndex((i) => (i + 1) % LOADING_PHASES.length), 1300);
    return () => clearInterval(interval);
  }, [active]);
  return LOADING_PHASES[phaseIndex];
}

export function ResearchAssistant() {
  const { patientId } = useParams<{ patientId: string }>();
  const [queryInput, setQueryInput] = useState("");
  const [asking, setAsking] = useState(false);
  const [answer, setAnswer] = useState<GroundedAnswer | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const loadingPhase = useLoadingPhase(asking);
  const evidenceCount = useScopedQuery(patientId ? `assistant-evidence-count-${patientId}` : null, () =>
    getPatientEvidence(patientId!, 1, 1)
  );

  // Patient isolation: switching patients clears any previous patient's
  // question/answer immediately.
  useEffect(() => {
    setQueryInput("");
    setAsking(false);
    setAnswer(null);
    setError(null);
  }, [patientId]);

  if (!patientId) return null;

  async function submitQuery(query: string) {
    const trimmed = query.trim();
    if (!trimmed || !patientId) return;

    setAsking(true);
    setError(null);
    setAnswer(null);
    try {
      const result = await askPatientQuestion(patientId, trimmed);
      setAnswer(result);
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError("network", "An unexpected error occurred."));
    } finally {
      setAsking(false);
    }
  }

  return (
    <div>
      <div className="breadcrumb">
        <Link to="/patients">Patients</Link> / <Link to={`/patients/${patientId}`}>{patientId}</Link> / Research assistant
      </div>

      <div className="app-topbar">
        <div className="page-header-text" style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div className="app-brand-mark" style={{ width: 40, height: 40 }}>
            <IconAssistant style={{ color: "#04121a" }} width={20} height={20} />
          </div>
          <div>
            <h1>Research Assistant</h1>
            <p>Evidence-grounded clinical intelligence</p>
          </div>
        </div>
      </div>

      <div className="card card-padded" style={{ marginBottom: 18 }}>
        <div style={{ display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 10, marginBottom: 14 }}>
          <div className="pill-row">
            <span className="badge badge-primary">Patient: {patientId}</span>
            {evidenceCount.data && (
              <span className="badge badge-neutral">{evidenceCount.data.pagination.total} evidence items indexed</span>
            )}
          </div>
        </div>

        <label htmlFor="assistant-query" style={{ fontWeight: 600, fontSize: 13 }}>
          Ask a question
        </label>
        <textarea
          id="assistant-query"
          ref={inputRef}
          className="input"
          style={{ marginTop: 8, minHeight: 84, resize: "vertical", fontSize: 14.5 }}
          placeholder="e.g. What is the patient's most recent blood pressure reading?"
          value={queryInput}
          onChange={(e) => setQueryInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submitQuery(queryInput);
          }}
          disabled={asking}
        />

        <div className="pill-row" style={{ marginTop: 10 }}>
          {SUGGESTED_QUESTIONS.map((q) => (
            <button
              key={q}
              type="button"
              className="btn btn-ghost"
              style={{ fontSize: 12 }}
              disabled={asking}
              onClick={() => {
                setQueryInput(q);
                inputRef.current?.focus();
              }}
            >
              {q}
            </button>
          ))}
        </div>

        <div style={{ marginTop: 14, display: "flex", alignItems: "center", gap: 12 }}>
          <button type="button" className="btn btn-primary" onClick={() => submitQuery(queryInput)} disabled={asking || !queryInput.trim()}>
            {asking ? "Asking…" : "Ask"}
          </button>
          {asking && (
            <span style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12.5, color: "var(--text-muted)" }}>
              <span className="spinner" /> {loadingPhase}
            </span>
          )}
        </div>

        <p className="text-faint" style={{ marginTop: 12, fontSize: 11.5 }}>
          Answers are grounded strictly in this patient's recorded evidence. If evidence is insufficient, the assistant
          says so rather than guessing.
        </p>
      </div>

      {error && <ErrorBlock error={error} onRetry={() => submitQuery(queryInput)} />}

      {answer && <AnswerPanel answer={answer} />}
    </div>
  );
}

function AnswerPanel({ answer }: { answer: GroundedAnswer }) {
  const accentColor =
    answer.status === "answered" ? "var(--green)" : answer.status === "insufficient_evidence" ? "var(--amber)" : "var(--gray)";

  return (
    <div className="card" style={{ borderColor: "var(--border-strong)", boxShadow: `0 0 32px -12px ${accentColor === "var(--green)" ? "var(--green-glow)" : "transparent"}` }}>
      <div className="card-padded" style={{ paddingBottom: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
          <div>
            <div className="text-faint" style={{ fontSize: 10.5, textTransform: "uppercase", letterSpacing: "0.06em" }}>Question</div>
            <div style={{ fontWeight: 600, marginTop: 3, fontSize: 14 }}>{answer.query}</div>
          </div>
          <AnswerStatusBadge status={answer.status} />
        </div>
      </div>

      <div className="card-section card-padded">
        <h3 style={{ color: accentColor }}>Answer</h3>

        {answer.status === "answered" && (
          <p style={{ fontSize: 14.5, lineHeight: 1.7, marginTop: 10 }}>{answer.answer_text}</p>
        )}

        {answer.status === "insufficient_evidence" && (
          <div style={{ marginTop: 10, padding: "14px 16px", background: "var(--amber-surface)", border: "1px solid var(--amber-border)", borderRadius: "var(--radius-md)" }}>
            <p style={{ fontSize: 13.5, color: "var(--amber)", fontWeight: 600 }}>
              There is insufficient recorded evidence to answer this question.
            </p>
            <p className="text-muted" style={{ fontSize: 12.5, marginTop: 6 }}>
              This is not a statement that the patient does or does not have any particular condition — it means the
              retrieval system could not find or verify a grounded answer in the patient's recorded evidence.
            </p>
          </div>
        )}

        {answer.status === "unsupported" && (
          <div style={{ marginTop: 10, padding: "14px 16px", background: "var(--gray-surface)", border: "1px solid var(--gray-border)", borderRadius: "var(--radius-md)" }}>
            <p style={{ fontSize: 13.5, color: "var(--text)", fontWeight: 600 }}>
              This question is outside the currently supported evidence domains.
            </p>
            <p className="text-muted" style={{ fontSize: 12.5, marginTop: 6 }}>
              The system could not identify a supported clinical concept or search term in this question.
            </p>
          </div>
        )}
      </div>

      {answer.status === "answered" && (
        <div className="card-section card-padded">
          <h3>Sources</h3>
          <div style={{ marginTop: 10 }}>
            <EvidenceCardList evidence={answer.evidence} />
          </div>
        </div>
      )}
    </div>
  );
}
