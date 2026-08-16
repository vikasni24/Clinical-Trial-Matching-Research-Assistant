"""The provider-independent LLM interface a future LLM-calling component
will depend on.

NO concrete provider is implemented here, and none is implemented anywhere
in the project yet — this defines only the shape: `generate(prompt: str)
-> str`. A later Phase 6D+ may add a real implementation (calling an
actual model, local or hosted), but it must conform to exactly this
Protocol so the rest of the application never depends on a specific
vendor SDK, API key, or network transport — callers depend on
LLMProvider alone, never on OpenAI/Anthropic/Gemini/etc. directly.

This module makes NO network calls, requires NO API key/credentials, and
imports nothing beyond the standard library. A deterministic fake
implementation for tests lives in tests/test_llm_provider.py, not here —
this file stays interface-only.
"""

from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    """A future text-generation provider: prompt in, generated text out.
    No retries, no streaming, no model-specific parameters, no API key —
    those are a concrete implementation's concern, not this interface's."""

    def generate(self, prompt: str) -> str: ...
