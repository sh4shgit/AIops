"""Centralised configuration loaded from environment variables.

Uses pydantic-settings so every value can be overridden via .env or the
process environment. Defaults match development/demo usage.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    from pydantic import BaseModel as _BaseModel

    class SettingsConfigDict(dict):  # type: ignore
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

    class BaseSettings(_BaseModel):  # type: ignore
        def __init__(self, **kwargs):
            # Read from os.environ as fallback if pydantic-settings is not installed
            env_vals = {}
            for field_name, field_info in self.__class__.model_fields.items():
                val = os.getenv(field_name.upper()) or os.getenv(field_name.lower())
                if val is not None:
                    if isinstance(field_info.default, SecretStr):
                        env_vals[field_name] = SecretStr(val)
                    elif isinstance(field_info.default, bool):
                        env_vals[field_name] = val.lower() in ("true", "1", "yes")
                    elif isinstance(field_info.default, int):
                        env_vals[field_name] = int(val)
                    elif isinstance(field_info.default, float):
                        env_vals[field_name] = float(val)
                    else:
                        env_vals[field_name] = val
            env_vals.update(kwargs)
            super().__init__(**env_vals)


class VectorStoreBackend(str, Enum):
    chromadb = "chromadb"
    pgvector = "pgvector"


class Settings(BaseSettings):
    """Application-wide settings, read once at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM -----------------------------------------------------------------
    anthropic_api_key: SecretStr = SecretStr("")
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.1

    # --- Redis ----------------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    redis_stream_name: str = "logs:incoming"
    redis_consumer_group: str = "aiops-workers"
    redis_dlq_stream: str = "logs:dead-letter"
    redis_max_retries: int = 3

    # --- Vector Store ---------------------------------------------------------
    vector_store_backend: VectorStoreBackend = VectorStoreBackend.chromadb
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "incidents"
    pgvector_url: str = "postgresql://user:password@localhost:5432/aiops"

    # --- Embedding ------------------------------------------------------------
    embedding_model: str = "all-MiniLM-L6-v2"
    retrieval_top_k: int = 5

    # --- Detection ------------------------------------------------------------
    anomaly_high_threshold: float = -0.3
    anomaly_low_threshold: float = -0.1
    isolation_forest_contamination: float = 0.05
    isolation_forest_n_estimators: int = 200
    feature_window_seconds: int = 300
    model_artifact_dir: str = "./data/models"

    # --- Actions --------------------------------------------------------------
    dry_run: bool = True
    jira_url: str = ""
    jira_user: str = ""
    jira_api_token: SecretStr = SecretStr("")
    jira_project_key: str = "OPS"
    slack_bot_token: SecretStr = SecretStr("")
    slack_channel: str = "#alerts"

    # --- API ------------------------------------------------------------------
    api_key: SecretStr = SecretStr("changeme")
    rate_limit_per_minute: int = 60
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Observability --------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "json"
    prometheus_enabled: bool = True


def get_settings() -> Settings:
    """Return the application settings singleton."""
    return Settings()
