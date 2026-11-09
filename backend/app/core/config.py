"""Application settings, loaded from environment variables (and an optional .env file).

All environment-specific values live here; nothing else in the application reads os.environ.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = BACKEND_DIR.parent


def _env_files() -> tuple[str, ...] | None:
    """APP_ENV_FILE overrides the .env location; set it to an empty string to disable .env
    loading entirely (the test suite does this so it never reads real credentials)."""
    override = os.environ.get("APP_ENV_FILE")
    if override is not None:
        return (override,) if override else None
    return (str(REPO_DIR / ".env"), ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Real environment variables always take precedence over the .env file.
        env_file=_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Runtime -------------------------------------------------------------------------------
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = True

    # --- HTTP ----------------------------------------------------------------------------------
    # Comma-separated in the environment, e.g. CORS_ORIGINS=https://app.example.com
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)
    frontend_dir: Path | None = REPO_DIR / "frontend"
    trust_proxy_headers: bool = False

    # --- Database ------------------------------------------------------------------------------
    database_url: str = f"sqlite:///{(REPO_DIR / 'data' / 'app.db').as_posix()}"
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=10, ge=0)
    auto_migrate: bool = True

    # --- Auth ----------------------------------------------------------------------------------
    allow_registration: bool = True
    access_token_ttl_hours: int = Field(default=24, ge=1, le=24 * 30)
    password_min_length: int = Field(default=10, ge=8)

    # --- Rate limits (requests per minute, per process) ----------------------------------------
    rate_limit_auth_per_minute: int = Field(default=10, ge=1)
    rate_limit_chat_per_minute: int = Field(default=20, ge=1)
    rate_limit_upload_per_minute: int = Field(default=10, ge=1)

    # --- Uploads -------------------------------------------------------------------------------
    upload_dir: Path = REPO_DIR / "data" / "uploads"
    max_upload_mb: int = Field(default=20, ge=1, le=500)
    max_files_per_upload: int = Field(default=10, ge=1, le=100)
    max_pdf_pages: int = Field(default=500, ge=1)
    max_extracted_chars: int = Field(default=5_000_000, ge=1_000)

    # --- Ingestion -----------------------------------------------------------------------------
    chunk_size_words: int = Field(default=180, ge=20, le=2000)
    chunk_overlap_words: int = Field(default=30, ge=0)
    ingestion_workers: int = Field(default=1, ge=1, le=8)
    ingestion_stale_minutes: int = Field(default=30, ge=1)

    # --- Embeddings ----------------------------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = Field(default=384, ge=8)
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_device: str | None = None
    embedding_preload: bool = True

    # --- Retrieval -----------------------------------------------------------------------------
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    # Cosine-similarity floor. Calibrated for all-MiniLM-L6-v2 on ~180-word multi-topic chunks:
    # relevant questions scored 0.22-0.73, unrelated ones 0.02-0.13 (tests/integration/test_real_embeddings.py).
    # Re-calibrate if you change EMBEDDING_MODEL or chunk size.
    retrieval_min_score: float = Field(default=0.20, ge=-1.0, le=1.0)
    retrieval_strong_score: float = Field(default=0.55, ge=-1.0, le=1.0)

    # --- LLM (OpenRouter / any OpenAI-compatible endpoint) ------------------------------------
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "mistralai/mistral-small-3.2-24b-instruct"
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=800, ge=16)
    llm_timeout_seconds: float = Field(default=60.0, gt=0)
    llm_connect_timeout_seconds: float = Field(default=10.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=5)
    llm_app_name: str = "Agentic RAG Assistant"
    llm_app_url: str | None = None

    # --- Agent ---------------------------------------------------------------------------------
    agent_max_iterations: int = Field(default=4, ge=1, le=10)
    agent_max_tool_calls: int = Field(default=4, ge=1, le=10)
    agent_timeout_seconds: float = Field(default=90.0, gt=0)
    agent_token_budget: int = Field(default=12_000, ge=1_000)
    agent_history_messages: int = Field(default=6, ge=0, le=20)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [o.strip() for o in value.split(",") if o.strip()]
        return value

    @field_validator("frontend_dir", mode="before")
    @classmethod
    def _empty_frontend_dir(cls, value: object) -> object:
        return None if value in ("", "none", "None") else value

    @model_validator(mode="after")
    def _check_consistency(self) -> Settings:
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError("CHUNK_OVERLAP_WORDS must be smaller than CHUNK_SIZE_WORDS")
        if self.retrieval_strong_score < self.retrieval_min_score:
            raise ValueError("RETRIEVAL_STRONG_SCORE must be >= RETRIEVAL_MIN_SCORE")
        if self.app_env == "production":
            if "*" in self.cors_origins:
                raise ValueError("CORS_ORIGINS must not contain '*' in production")
            if self.database_url.startswith("sqlite"):
                raise ValueError("Use PostgreSQL (DATABASE_URL=postgresql+psycopg://...) in production")
        return self

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def llm_configured(self) -> bool:
        return bool(self.openrouter_api_key and self.openrouter_api_key.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
