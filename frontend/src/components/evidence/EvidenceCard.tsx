import type { Evidence } from "../../types/api";
import { formatDate, formatEvidenceValue } from "../../utils/format";

/**
 * Renders one Evidence item as a readable card — resource type, code,
 * display, value/unit, effective date, status, and a clear traceability
 * line ("Source: Observation/{resource_id}"). Never renders raw FHIR JSON;
 * only the already-extracted Evidence fields the backend returned.
 */
export function EvidenceCard({ evidence }: { evidence: Evidence }) {
  const sourceReference = evidence.source_reference ?? `${evidence.resource_type}/${evidence.resource_id}`;

  return (
    <div
      style={{
        border: "1px solid var(--border)",
        borderRadius: "var(--radius-md)",
        padding: "12px 14px",
        background: "var(--surface)",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
        <span className="badge badge-primary">{evidence.resource_type}</span>
        {evidence.status && <span className="text-faint">{evidence.status}</span>}
      </div>

      <div style={{ marginTop: 8, fontWeight: 600 }}>{evidence.display ?? "Unnamed finding"}</div>

      <div style={{ marginTop: 6, display: "flex", flexWrap: "wrap", gap: "6px 18px", fontSize: 12.5 }}>
        {evidence.value !== null && evidence.value !== undefined && (
          <span>
            <span className="text-faint">Value: </span>
            {formatEvidenceValue(evidence.value, evidence.unit)}
          </span>
        )}
        {evidence.code && (
          <span>
            <span className="text-faint">Code: </span>
            <span className="text-mono">{evidence.code}</span>
          </span>
        )}
        {evidence.effective_date && (
          <span>
            <span className="text-faint">Date: </span>
            {formatDate(evidence.effective_date)}
          </span>
        )}
      </div>

      <div style={{ marginTop: 8, fontSize: 12, color: "var(--text-faint)" }}>
        Source: <span className="text-mono">{sourceReference}</span>
      </div>
    </div>
  );
}

export function EvidenceCardList({ evidence }: { evidence: Evidence[] }) {
  if (evidence.length === 0) {
    return <p className="text-muted" style={{ fontSize: 13 }}>No evidence available.</p>;
  }
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: 10 }}>
      {evidence.map((item) => (
        <EvidenceCard key={`${item.resource_type}/${item.resource_id}`} evidence={item} />
      ))}
    </div>
  );
}
