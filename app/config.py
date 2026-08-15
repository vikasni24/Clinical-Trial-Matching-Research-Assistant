"""Application configuration loaded from environment variables / .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_database: str = "clinical_trial_matching"
    synthea_fhir_dir: str = "data/synthea/fhir"
    trials_data_path: str = "data/trials/dev_trials.json"

    default_page_size: int = 20
    max_page_size: int = 100

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
