"""Phase 6G: orchestrates the grounded-answer pipeline end to end for
POST /api/patients/{patient_id}/ask. Phase 7 adds a deterministic audit
record of the outcome, written after the fact.

    patient_id, query
          |
    HybridEvidenceRetriever   (via GroundedContextService, Phase 5E)
          |
     GroundedContext
          |
  enforce_pre_generation_safety()   (Phase 6F)
      /              \
GroundedAnswer   build_grounded_prompt()   (Phase 6B)
(short-circuit,        |
 no LLM call)   LLMProvider.generate()     (Phase 6C/6D)
                        |
                build_grounded_answer()    (Phase 6E)
                        |
                   GroundedAnswer
                        |
           AuditService.record_question_audit()   (Phase 7)
                        |
                   GroundedAnswer   (returned to the caller, unchanged)

This module performs no retrieval, prompting, generation, or validation
logic of its own — it only wires together the already-built, already-tested
components from Phases 5E and 6A-6F, in the exact order the endpoint
requires. It never queries MongoDB directly (that stays inside
GroundedContextService/HybridEvidenceRetriever/AuditService) and never
imports a concrete LLM provider (it depends on the LLMProvider protocol
only, so the caller decides which implementation — real or fake — to
inject). Patient existence is validated by the caller (see
app/api/routes/patients.py) before this service is ever invoked.

AUDIT PLACEMENT (Phase 7E): the audit write happens strictly AFTER
build_grounded_answer() (the Phase 6E validation step) or after
enforce_pre_generation_safety()'s short-circuit — i.e. only once a final,
fully-validated GroundedAnswer exists. It is never possible for an audit
record to describe an answer that hasn't already passed validation, because
there is no code path that constructs one from anything other than that
already-validated GroundedAnswer.

AUDIT FAILURE ISOLATION (Phase 7E): audit persistence is best-effort and
deliberately isolated from the answer path. `_record_audit_safely` catches
any exception audit persistence raises (e.g. a MongoDB write failure) and
only logs it — it never re-raises, never falls back to returning raw LLM
text, and never alters `answer` in any way. The exact same, already-safe
GroundedAnswer is returned to the caller whether or not the audit write
succeeded. Conversely, if the LLM call itself fails (LLMProviderError), no
GroundedAnswer was ever produced, so no audit record is written for that
request at all — the exception propagates unchanged to the caller (see
app/api/routes/patients.py's existing error handling), exactly as it did
before Phase 7.
"""

from __future__ import annotations

import logging

from pymongo.database import Database

from app.models.answer import GroundedAnswer
from app.models.rag_context import GroundedContext
from app.models.retrieval import RetrievalRequest
from app.services.answer_validator import build_grounded_answer
from app.services.audit_service import AuditService
from app.services.grounded_prompt import build_grounded_prompt
from app.services.llm_provider import LLMProvider
from app.services.rag_context import GroundedContextService
from app.services.safety_rules import enforce_pre_generation_safety

logger = logging.getLogger(__name__)


class AskService:
    def __init__(self, db: Database, llm_provider: LLMProvider):
        self._context_service = GroundedContextService(db)
        self._llm_provider = llm_provider
        self._audit_service = AuditService(db)

    def ask(self, patient_id: str, query: str) -> GroundedAnswer:
        context = self._context_service.build_context(RetrievalRequest(patient_id=patient_id, query=query))

        gated_answer = enforce_pre_generation_safety(context)
        if gated_answer is not None:
            self._record_audit_safely(context, gated_answer)
            return gated_answer

        prompt = build_grounded_prompt(context)
        prompt_text = "\n\n".join(
            [prompt.instructions, prompt.status_note, f"QUESTION: {prompt.query}", prompt.evidence_text]
        )
        raw_answer_text = self._llm_provider.generate(prompt_text)

        answer = build_grounded_answer(context, raw_answer_text)
        self._record_audit_safely(context, answer)
        return answer

    def _record_audit_safely(self, context: GroundedContext, answer: GroundedAnswer) -> None:
        """Persists an audit record for an already-validated answer.
        Deliberately broad exception handling: this is the one sanctioned
        boundary where a secondary, best-effort concern (audit persistence)
        must never be allowed to affect, delay, or replace the primary
        response path. Only high-level facts (patient_id, the exception
        itself) are logged — never the raw LLM prompt or generated text."""
        try:
            self._audit_service.record_question_audit(context, answer)
        except Exception as exc:  # noqa: BLE001 — intentional: see docstring above
            logger.warning("Failed to persist audit record for patient '%s': %s", context.patient_id, exc)
