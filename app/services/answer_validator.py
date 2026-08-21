"""Deterministic post-processing/validation: turns a raw LLM-generated
answer string into a GroundedAnswer (Phase 6A), enforcing that every claim
the answer relies on is traceable to Evidence that was actually supplied
to the generation process. The LLM's output is never trusted just because
it was returned successfully — it is validated against the same Evidence
the prompt was built from before being allowed to become an "answered"
GroundedAnswer.

    GroundedContext + raw LLM text
              |
      build_grounded_answer()
              |
         GroundedAnswer

This is intentionally NOT a medical fact-checker: it does not evaluate
whether the answer's clinical content is correct, only whether every
evidence reference it makes resolves to something that was genuinely
supplied. That is a citation/traceability check, not clinical reasoning —
deterministic and contract-based, per this phase's scope.

DECISION RULE (deterministic, evaluated in this precedence):
  1. No evidence was ever supplied (context.status != "evidence_found")
     -> the raw answer text is never trusted; the result mirrors the
        underlying no-evidence/unsupported state.
  2. The answer cites no evidence at all
     -> insufficient_evidence (an answer that cites nothing cannot be
        verified as grounded, even if evidence was available).
  3. The answer cites at least one [ResourceType/resource_id] that was NOT
     among the supplied evidence (a fabricated/invented reference)
     -> insufficient_evidence — never silently drop the fabrication and
        answer anyway.
  4. Every citation resolves to real, supplied evidence
     -> answered, with GroundedAnswer.evidence set to exactly the
        supplied Evidence objects that were actually cited (a subset of
        context.evidence, in its original order — never a newly
        constructed or reconstructed Evidence object).
"""

from __future__ import annotations

import re

from app.models.answer import GroundedAnswer
from app.models.evidence import Evidence
from app.models.rag_context import GroundedContext

# Matches exactly the citation format app.services.grounded_prompt's
# _format_evidence uses to present evidence to the model, and that the
# SAFETY_INSTRUCTIONS tell the model to cite claims with:
# "[ResourceType/resource_id]".
_CITATION_PATTERN = re.compile(r"\[([A-Za-z][A-Za-z0-9]*)/([^\[\]/]+)\]")


def _extract_cited_keys(answer_text: str) -> set[tuple[str, str]]:
    """Deterministically extracts every (resource_type, resource_id) pair
    cited in the answer text. Pure regex matching — no semantic
    interpretation, no LLM involvement."""
    return {(resource_type, resource_id) for resource_type, resource_id in _CITATION_PATTERN.findall(answer_text)}


def build_grounded_answer(context: GroundedContext, raw_answer_text: str) -> GroundedAnswer:
    """Deterministically validates `raw_answer_text` (as produced by an
    LLMProvider from a prompt built from `context`, see
    app/services/grounded_prompt.py) against the Evidence `context`
    actually supplied, and returns a GroundedAnswer that can never claim
    more than what was genuinely grounded."""

    # Rule 1: nothing was ever supplied to ground an answer in — the raw
    # text is not even inspected.
    if context.status != "evidence_found" or not context.evidence:
        status = "unsupported" if context.status == "unsupported" else "insufficient_evidence"
        return GroundedAnswer(
            patient_id=context.patient_id,
            query=context.query,
            status=status,
            message=context.message or "No evidence was available to ground an answer.",
            retrieval_status=context.status,
        )

    cited_keys = _extract_cited_keys(raw_answer_text)

    # Rule 2: an uncited answer cannot be verified as grounded.
    if not cited_keys:
        return GroundedAnswer(
            patient_id=context.patient_id,
            query=context.query,
            status="insufficient_evidence",
            message="Generated answer did not cite any supplied evidence and cannot be verified as grounded.",
            retrieval_status=context.status,
        )

    available_keys = {(item.resource_type, item.resource_id) for item in context.evidence}
    fabricated_keys = cited_keys - available_keys

    # Rule 3: any citation not among the supplied evidence is a fabrication.
    if fabricated_keys:
        return GroundedAnswer(
            patient_id=context.patient_id,
            query=context.query,
            status="insufficient_evidence",
            message=(
                "Generated answer referenced evidence that was not supplied "
                f"({len(fabricated_keys)} fabricated reference(s)); rejected rather than trusted."
            ),
            retrieval_status=context.status,
        )

    # Rule 4: every citation resolves — pull the ACTUAL supplied Evidence
    # objects (never construct new ones), preserving their original order.
    matched_evidence: list[Evidence] = [
        item for item in context.evidence if (item.resource_type, item.resource_id) in cited_keys
    ]

    # Defense in depth: GroundedAnswer's own validator already enforces
    # this, but an explicit, loud check here makes a violation immediately
    # attributable to this layer rather than surfacing as an opaque
    # downstream ValidationError.
    mismatched = [item for item in matched_evidence if item.patient_id != context.patient_id]
    if mismatched:
        raise ValueError(
            f"build_grounded_answer received evidence for patient '{mismatched[0].patient_id}' while grounding "
            f"an answer for patient '{context.patient_id}' — cross-patient evidence is not permitted"
        )

    return GroundedAnswer(
        patient_id=context.patient_id,
        query=context.query,
        status="answered",
        answer_text=raw_answer_text,
        evidence=matched_evidence,
        retrieval_status=context.status,
    )
