"""Application configuration loaded from environment variables / .env file."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "clinical_trial_matching"
    synthea_fhir_dir: str = "data/synthea/fhir"
    trials_data_path: str = "data/trials/dev_trials.json"

    default_page_size: int = 20
    max_page_size: int = 100

    # Frontend Phase 1: origins allowed to call this API via CORS (see
    # app/main.py). Defaults cover Vite's default dev server ports only —
    # never a wildcard, and never something that also allows credentials
    # from an untrusted origin.
    cors_allowed_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # LLM provider configuration (Phase 6D; multi-provider support added
    # later). llm_api_key has no default — it must come from the
    # environment and is never hardcoded. The concrete provider
    # (app/services/anthropic_llm_provider.py or
    # app/services/groq_llm_provider.py) fails fast at construction time
    # if it's missing. llm_provider selects which one get_llm_provider()
    # (app/api/routes/patients.py) constructs — "anthropic" (default) or
    # "groq". llm_model has no default here: each concrete provider
    # supplies its own vendor-appropriate default when unset, since a
    # single hardcoded default couldn't be right for both vendors.
    llm_provider: str = "anthropic"
    llm_api_key: Optional[str] = None
    llm_model: Optional[str] = None
    llm_timeout_seconds: float = 30.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
