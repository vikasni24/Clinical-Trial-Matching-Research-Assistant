import { useState } from "react";
import type { CriterionEvaluationOut } from "../../types/api";
import { EligibilityResultBadge } from "../common/StatusBadge";
import { EvidenceCardList } from "../evidence/EvidenceCard";
import { formatValue } from "../../utils/format";

/**
 * One eligibility criterion row. UNKNOWN criteria always show the explicit
 * "insufficient information" framing required by this phase — never
 * rendered or implied as a confident negative. FAIL criteria show the
 * backend's own `reason` text verbatim (already a deterministic,
 * evidence-grounded explanation from EligibilityMatcher) rather than any
 * UI-invented clinical language.
 */
export function CriterionRow({ criterion }: { criterion: CriterionEvaluationOut }) {
  const [expanded, setExpanded] = useState(false);
  const hasEvidence = criterion.evidence.length > 0;

  return (
    <div style={{ borderBottom: "1px solid var(--border)", padding: "12px 0" }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "flex-start" }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <EligibilityResultBadge result={criterion.result} />
            <span style={{ fontWeight: 600 }}>{criterion.criterion}</span>
          </div>

          {criterion.result === "UNKNOWN" ? (
            <p className="text-muted" style={{ marginTop: 6, fontSize: 13 }}>
              Insufficient information to determine this criterion.{" "}
              <span className="text-faint">({criterion.reason})</span>
            </p>
          ) : (
            <p className="text-muted" style={{ marginTop: 6, fontSize: 13 }}>
              {criterion.reason}
            </p>
          )}

          <div className="pill-row" style={{ marginTop: 6, fontSize: 12 }}>
            {criterion.required_value && (
              <span className="text-faint">Required: {criterion.required_value}</span>
            )}
            {criterion.patient_value !== null && criterion.patient_value !== undefined && (
              <span className="text-faint">Patient value: {formatValue(criterion.patient_value)}</span>
            )}
          </div>
        </div>

        {hasEvidence && (
          <button type="button" className="btn btn-secondary" onClick={() => setExpanded((e) => !e)}>
            {expanded ? "Hide evidence" : `View evidence (${criterion.evidence.length})`}
          </button>
        )}
      </div>

      {expanded && hasEvidence && (
        <div style={{ marginTop: 10 }}>
          <EvidenceCardList evidence={criterion.evidence} />
        </div>
      )}
    </div>
  );
}
