"""A second concrete LLMProvider (see app/services/llm_provider.py's
Protocol) implementation: a raw HTTP client for Groq's OpenAI-compatible
chat completions API. Uses only the `httpx` dependency already present in
this project (the same one AnthropicLLMProvider uses) — no vendor SDK is
added, and no new dependency was introduced to support this provider.

Reuses AnthropicLLMProvider's LLMProviderError / LLMProviderConfigurationError
/ LLMProviderRequestError exception hierarchy (defined there, imported here
rather than redefined) so callers — AskService, the /ask route — can catch
provider failures uniformly regardless of which concrete provider is
active. Which provider is active is controlled entirely by
Settings.llm_provider (see app/config.py) and selected in
app/api/routes/patients.py's get_llm_provider(); AskService itself never
imports this module or knows which vendor is behind the LLMProvider it was
given.

Architecture boundary — identical to AnthropicLLMProvider:
  - It receives an already-built prompt string (see
    app/services/grounded_prompt.py) and returns generated text only.
  - It never queries MongoDB, never imports EvidenceService, the evidence
    repository, or HybridEvidenceRetriever, and therefore can never see a
    raw FHIR document or perform its own retrieval.
  - It has no autonomous tool use, no browsing, no hidden context, and
    stores no chain-of-thought — `generate()` returns exactly the text the
    provider responded with, nothing more.

Configuration (API key, model, timeout) comes exclusively from
app.config.Settings / environment variables — never hardcoded.
"""

from __future__ import annotations

from typing import Optional

import httpx

from app.config import Settings, get_settings
from app.services.anthropic_llm_provider import LLMProviderConfigurationError, LLMProviderRequestError

_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# "llama-3.3-70b-versatile" (the original default) has since been removed
# from Groq's free-tier model catalog for at least some accounts/keys —
# confirmed live via a 404 model_not_found response. openai/gpt-oss-20b is
# currently available and, importantly, is a "reasoning" model that keeps
# its internal reasoning in a separate `reasoning` field (never read by
# this provider — see generate() below) and its final answer in the
# standard `content` field, same as any other OpenAI-compatible model.
_DEFAULT_MODEL = "openai/gpt-oss-20b"
# Reasoning models spend part of their token budget on internal reasoning
# before writing `content` — 1024 was frequently not enough for the model
# to finish reasoning AND produce an answer (content came back empty,
# finish_reason="length"). 2048 was confirmed sufficient in live testing.
_MAX_TOKENS = 2048


class GroqLLMProvider:
    """Satisfies app.services.llm_provider.LLMProvider by structural
    typing. `client` is injectable so tests can supply an httpx.Client
    backed by httpx.MockTransport — no real network call is ever needed
    to exercise this class's request-building/response-parsing logic."""

    def __init__(self, settings: Optional[Settings] = None, client: Optional[httpx.Client] = None):
        settings = settings or get_settings()
        if not settings.llm_api_key:
            raise LLMProviderConfigurationError(
                "LLM_API_KEY is not configured — set it via environment variable before using GroqLLMProvider"
            )
        self._api_key = settings.llm_api_key
        self._model = settings.llm_model or _DEFAULT_MODEL
        self._timeout = settings.llm_timeout_seconds
        self._client = client or httpx.Client()

    def generate(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("prompt must not be empty")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
        }
        payload = {
            "model": self._model,
            "max_tokens": _MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            response = self._client.post(_GROQ_API_URL, headers=headers, json=payload, timeout=self._timeout)
        except httpx.TimeoutException as exc:
            raise LLMProviderRequestError(f"LLM provider request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMProviderRequestError(f"LLM provider network error: {exc}") from exc

        if response.status_code != 200:
            raise LLMProviderRequestError(
                f"LLM provider returned an error response: {response.status_code} {response.text}"
            )

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMProviderRequestError(f"LLM provider returned an unexpected response shape: {exc}") from exc
