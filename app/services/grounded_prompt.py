"""Deterministic conversion of GroundedContext (Phase 5E) into a strict,
LLM-ready prompt payload — CONTRACT/BUILDER ONLY. This module never calls
an LLM, never generates an answer, and never adds a scoring/reasoning
field; it only assembles a fixed set of safety instructions plus a
deterministic, traceable rendering of the already-retrieved Evidence.

    GroundedContext
          |
    build_grounded_prompt()
          |
     GroundedPrompt
          |
   (a later Phase 6 sub-phase: an actual LLM call — NOT implemented here)

Every field on GroundedPrompt is either a fixed instruction template or a
direct, deterministic rendering of Evidence already present in the
GroundedContext — nothing is generated, inferred, or fetched here.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.models.evidence import Evidence
from app.models.rag_context import GroundedContext
from app.models.retrieval import RetrievalStatus

# The fixed safety instructions every GroundedPrompt carries verbatim.
# Each numbered rule maps directly to a required directive: (1) evidence-
# only grounding, (2) no invented facts, (3) absence-of-evidence is not a
# negative fact, (4) explicit insufficiency rather than guessing, (5) no
# raw FHIR exposure, (6) citeable traceability, (7) patient isolation,
# (8) no fabricated citations, (9) no unsupported certainty.
SAFETY_INSTRUCTIONS = (
    "You are a clinical evidence assistant. Follow these rules strictly:\n"
    "1. Use ONLY the evidence supplied below. Do not use any other knowledge about this patient.\n"
    "2. Do not invent, guess, or infer any patient fact that is not explicitly present in the supplied evidence.\n"
    "3. Do not infer that missing evidence means the patient does not have a condition, medication, allergy, "
    "or lab value. Absence of evidence is not evidence of absence.\n"
    "4. If the supplied evidence is insufficient to answer the query, say so explicitly rather than guessing.\n"
    "5. Do not describe, quote, or expose raw FHIR documents or internal database structure — only the "
    "evidence fields provided below.\n"
    "6. Every claim in your answer must be traceable to a specific evidence item by its resource_type and "
    "resource_id.\n"
    "7. Never use or reference evidence belonging to a patient other than the one specified below.\n"
    "8. Do not fabricate citations, resource IDs, or evidence that was not supplied.\n"
    "9. Do not state or imply certainty beyond what the supplied evidence actually supports."
)

_NO_EVIDENCE_NOTE = (
    "STATUS: No matching evidence was found in the patient's record for this query. "
    "You must explicitly state that the evidence is insufficient to answer — never guess or infer an answer."
)
_UNSUPPORTED_NOTE = (
    "STATUS: This query could not be processed by the retrieval system. "
    "You must explicitly state that you cannot answer this query with the available tools — never guess."
)
_EVIDENCE_FOUND_NOTE = "STATUS: Evidence was found for this patient and is provided below."

_STATUS_NOTES: dict[RetrievalStatus, str] = {
    "evidence_found": _EVIDENCE_FOUND_NOTE,
    "no_evidence_found": _NO_EVIDENCE_NOTE,
    "unsupported": _UNSUPPORTED_NOTE,
}


class GroundedPrompt(BaseModel):
    """A deterministic, LLM-ready payload. `instructions` + `status_note` +
    `evidence_text` together form the text a future LLM call would send;
    `evidence` is the same information in structured form, for citation
    and traceability checks."""

    patient_id: str
    query: str
    status: RetrievalStatus
    instructions: str
    status_note: str
    evidence_text: str
    evidence: list[Evidence] = []


def _format_evidence(evidence: list[Evidence]) -> str:
    """Deterministic, citeable, flat-Evidence-only rendering — never raw
    FHIR, never MongoDB _id, never anything not already on Evidence."""
    if not evidence:
        return "No evidence was supplied."

    lines: list[str] = []
    for item in evidence:
        pieces: list[str] = []
        if item.display:
            pieces.append(item.display)
        if item.code:
            pieces.append(f"code={item.code}")
        if item.value is not None:
            value_piece = f"value={item.value}"
            if item.unit:
                value_piece += f" {item.unit}"
            pieces.append(value_piece)
        if item.effective_date:
            pieces.append(f"date={item.effective_date}")
        if item.status:
            pieces.append(f"status={item.status}")
        descriptor = ", ".join(pieces) if pieces else "(no additional detail recorded)"
        lines.append(f"[{item.resource_type}/{item.resource_id}] {descriptor}")
    return "\n".join(lines)


def build_grounded_prompt(context: GroundedContext) -> GroundedPrompt:
    """Deterministically converts a GroundedContext into a GroundedPrompt.
    Pure data transformation — no LLM call, no additional retrieval, no
    generation. The same GroundedContext always produces the same
    GroundedPrompt."""
    return GroundedPrompt(
        patient_id=context.patient_id,
        query=context.query,
        status=context.status,
        instructions=SAFETY_INSTRUCTIONS,
        status_note=_STATUS_NOTES[context.status],
        evidence_text=_format_evidence(context.evidence),
        evidence=context.evidence,
    )
