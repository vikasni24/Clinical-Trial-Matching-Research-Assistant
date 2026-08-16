import type { AnswerStatus, EligibilityResult, OverallEligibilityStatus, RetrievalStatus } from "../../types/api";

type BadgeTone = "success" | "danger" | "warning" | "neutral" | "primary";

function Badge({ tone, label }: { tone: BadgeTone; label: string }) {
  return <span className={`badge badge-${tone}`}>{label}</span>;
}

/** Renders one of the backend's PASS/FAIL/UNKNOWN criterion results.
 * UNKNOWN is never rendered as, or confused with, FAIL. */
export function EligibilityResultBadge({ result }: { result: EligibilityResult }) {
  switch (result) {
    case "PASS":
      return <Badge tone="success" label="PASS" />;
    case "FAIL":
      return <Badge tone="danger" label="FAIL" />;
    case "UNKNOWN":
      return <Badge tone="warning" label="UNKNOWN" />;
  }
}

export function OverallEligibilityBadge({ status }: { status: OverallEligibilityStatus }) {
  switch (status) {
    case "ELIGIBLE":
      return <Badge tone="success" label="Eligible" />;
    case "INELIGIBLE":
      return <Badge tone="danger" label="Ineligible" />;
    case "UNKNOWN":
      return <Badge tone="warning" label="Unknown" />;
  }
}

/** Renders one of the backend's answered/insufficient_evidence/unsupported
 * GroundedAnswer states. insufficient_evidence is never rendered as a
 * negative clinical finding. */
export function AnswerStatusBadge({ status }: { status: AnswerStatus }) {
  switch (status) {
    case "answered":
      return <Badge tone="success" label="Answered" />;
    case "insufficient_evidence":
      return <Badge tone="warning" label="Insufficient evidence" />;
    case "unsupported":
      return <Badge tone="neutral" label="Unsupported query" />;
  }
}

export function RetrievalStatusBadge({ status }: { status: RetrievalStatus }) {
  switch (status) {
    case "evidence_found":
      return <Badge tone="success" label="Evidence found" />;
    case "no_evidence_found":
      return <Badge tone="warning" label="No evidence found" />;
    case "unsupported":
      return <Badge tone="neutral" label="Unsupported" />;
  }
}
