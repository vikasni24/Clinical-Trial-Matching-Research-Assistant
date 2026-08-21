"""The one concrete LLMProvider (Phase 6C protocol) implementation: a raw
HTTP client for Anthropic's Messages API. Uses only the `httpx` dependency
already present in this project (added in Phase 1 for the FastAPI test
client) — no vendor SDK is added.

Architecture boundary — this class is a pure text-in/text-out adapter:
  - It receives an already-built prompt string (see
    app/services/grounded_prompt.py) and returns generated text only.
  - It never queries MongoDB, never imports EvidenceService, the evidence
    repository, or HybridEvidenceRetriever, and therefore can never see a
    raw FHIR document or perform its own retrieval — building the
    grounded prompt is entirely the caller's responsibility, before this
    class is ever invoked.
  - It has no autonomous tool use, no browsing, no hidden context, and
    stores no chain-of-thought — `generate()` returns exactly the text the
    provider responded with, nothing more.

Configuration (API key, model, timeout) comes exclusively from
app.config.Settings / environment variables — never hardcoded. Missing
configuration, network failures, timeouts, and non-success provider
responses are all normalized into one small exception hierarchy so
callers have a single, deterministic boundary to handle.
"""

from __future__ import annotations

from typing import Optional

import httpx

from app.config import Settings, get_settings

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_API_VERSION = "2023-06-01"
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 1024


class LLMProviderError(Exception):
    """Base class for all LLM provider failures at the application boundary."""


class LLMProviderConfigurationError(LLMProviderError):
    """Raised when required configuration (e.g. the API key) is missing."""


class LLMProviderRequestError(LLMProviderError):
    """Raised when the underlying request fails: network error, timeout,
    or a non-success/unexpected response from the provider."""


class AnthropicLLMProvider:
    """Satisfies app.services.llm_provider.LLMProvider by structural
    typing. `client` is injectable so tests can supply an httpx.Client
    backed by httpx.MockTransport — no real network call is ever needed
    to exercise this class's request-building/response-parsing logic."""

    def __init__(self, settings: Optional[Settings] = None, client: Optional[httpx.Client] = None):
        settings = settings or get_settings()
        if not settings.llm_api_key:
            raise LLMProviderConfigurationError(
                "LLM_API_KEY is not configured — set it via environment variable before using AnthropicLLMProvider"
            )
        self._api_key = settings.llm_api_key
        self._model = settings.llm_model or _DEFAULT_MODEL
        self._timeout = settings.llm_timeout_seconds
        self._client = client or httpx.Client()

    def generate(self, prompt: str) -> str:
        if not prompt:
            raise ValueError("prompt must not be empty")

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": self._model,
            "max_tokens": _MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }

        try:
            response = self._client.post(_ANTHROPIC_API_URL, headers=headers, json=payload, timeout=self._timeout)
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
            return data["content"][0]["text"]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMProviderRequestError(f"LLM provider returned an unexpected response shape: {exc}") from exc
