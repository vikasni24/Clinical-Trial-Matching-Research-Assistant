"""Phase 6F: pre-generation safety gate — hardens the boundary between
retrieved evidence (GroundedContext, Phase 5E) and an actual LLM call.

app/services/answer_validator.py (Phase 6E) validates an answer AFTER an
LLMProvider has already run. Without a gate in front of it, a
no-evidence/unsupported/cross-patient-contaminated GroundedContext would
still be turned into a prompt (app/services/grounded_prompt.py) and handed
to an LLMProvider before Phase 6E's post-hoc check ever ran — the unsafe
content would already have left the process. This module closes that gap by
deciding, deterministically and before any prompt is built or any
LLMProvider is invoked, whether generation should even be attempted.

    GroundedContext
          |
  enforce_pre_generation_safety()
      /        \
  GroundedAnswer   None
  (short-circuit,  (safe to build a prompt and call the LLM)
   no LLM call)

DETERMINISTIC SAFETY RULES enforced here (Phase 6F task list items 1, 2, 4):
  1. No evidence (status == "no_evidence_found")
     -> insufficient_evidence, immediately. No LLM call is made.
  2. Unsupported query (status == "unsupported")
     -> unsupported, immediately. No LLM call is made.
  4. Cross-patient evidence somehow present on the context (defense in
     depth against an upstream invariant violation bypassing
     GroundedContext's own validator)
     -> raise ValueError immediately, before any evidence content could
        reach a prompt or an LLM.

Task list items 3, 5, 6 are enforced downstream, unmodified, by
app/services/grounded_prompt.py (evidence-only rendering, never raw FHIR,
never a fabricated value for a missing field) and
app/services/answer_validator.py (an uncited or fabricated-citation answer
is rejected rather than trusted). Item 7 (UNKNOWN eligibility semantics) is
enforced, unmodified, by app/services/eligibility_matcher.py — this module
does not touch eligibility logic at all.
"""

from __future__ import annotations

from typing import Optional

from app.models.answer import GroundedAnswer
from app.models.rag_context import GroundedContext


def enforce_pre_generation_safety(context: GroundedContext) -> Optional[GroundedAnswer]:
    """Returns a GroundedAnswer immediately, without ever building a prompt
    or invoking an LLMProvider, when generation must not be attempted.
    Returns None when it is safe for the caller to proceed to
    build_grounded_prompt() -> LLMProvider.generate() ->
    answer_validator.build_grounded_answer()."""

    # Rule 4: cross-patient evidence — checked first and loudest, since it
    # is the one violation that must never even reach a prompt.
    mismatched = [item for item in context.evidence if item.patient_id != context.patient_id]
    if mismatched:
        raise ValueError(
            f"GroundedContext for patient '{context.patient_id}' contains evidence belonging to a "
            f"different patient ('{mismatched[0].patient_id}') — refusing to build a prompt or call an LLM"
        )

    # Rule 2: unsupported query.
    if context.status == "unsupported":
        return GroundedAnswer(
            patient_id=context.patient_id,
            query=context.query,
            status="unsupported",
            message=context.message or "This query could not be serviced by the retrieval system.",
        )

    # Rule 1: no evidence was found.
    if context.status == "no_evidence_found":
        return GroundedAnswer(
            patient_id=context.patient_id,
            query=context.query,
            status="insufficient_evidence",
            message=context.message or "No evidence was found for this patient to ground an answer.",
        )

    return None
