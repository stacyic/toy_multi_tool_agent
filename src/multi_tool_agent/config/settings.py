"""Pydantic settings for configuration management."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    """OpenAI-specific configuration."""

    model_config = SettingsConfigDict(
        env_prefix="OPENAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = Field(default="", description="OpenAI API key")
    chat_model: str = Field(default="gpt-4", description="Chat model for SQL generation and routing")
    embedding_model: str = Field(
        default="text-embedding-3-small", description="Embedding model for RAG"
    )
    temperature: float = Field(default=0.0, description="Temperature for deterministic responses")
    max_retries: int = Field(default=3, description="Max retries for API calls")


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DATABASE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    path: Path = Field(default=Path("data/store.db"), description="Path to SQLite database")


class LoggingSettings(BaseSettings):
    """Logging configuration."""

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    destination: Literal["console", "file", "both"] = Field(
        default="both", description="Log destination"
    )
    file: Path = Field(default=Path("logs/agent.log"), description="Log file path")
    level: str = Field(default="INFO", description="Log level")
    include_costs: bool = Field(default=True, description="Include cost tracking in logs")


class AgentSettings(BaseSettings):
    """Agent behavior configuration."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sql_max_retries: int = Field(default=3, description="Max SQL execution retries")
    policy_cache_ttl: int = Field(default=3600, description="Policy cache TTL in seconds")
    graceful_degradation: bool = Field(
        default=True, description="Enable graceful degradation on API failures"
    )


class SQLValidationSettings(BaseSettings):
    """SQL semantic validation and error correction configuration."""

    model_config = SettingsConfigDict(
        env_prefix="SQL_VALIDATION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Semantic evaluation settings
    enable_semantic_eval: bool = Field(
        default=False,
        description="Enable LLM-based semantic evaluation of SQL queries"
    )
    semantic_eval_model: str = Field(
        default="",
        description="Model for semantic evaluation (empty = use same as generator)"
    )
    semantic_confidence_threshold: float = Field(
        default=0.7,
        description="Minimum confidence score to accept a query as semantically correct"
    )

    # Human review settings
    enable_human_review: bool = Field(
        default=False,
        description="Enable recording ambiguous queries for human review"
    )
    review_queue_path: Path = Field(
        default=Path("data/review_queue.json"),
        description="Path to human review queue file"
    )
    max_review_queue_size: int = Field(
        default=1000,
        description="Maximum items to keep in review queue"
    )

    # Golden set evaluation settings
    golden_set_path: Path = Field(
        default=Path("data/golden_set.json"),
        description="Path to golden set evaluation data"
    )
    metrics_dir: Path = Field(
        default=Path("data/metrics"),
        description="Directory to store accuracy metrics"
    )


class Settings(BaseSettings):
    """Root configuration aggregating all settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    sql_validation: SQLValidationSettings = Field(default_factory=SQLValidationSettings)

    # Data paths
    policies_path: Path = Field(default=Path("data/policies.md"), description="Path to policies file")
    data_dir: Path = Field(default=Path("data"), description="Data directory")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
