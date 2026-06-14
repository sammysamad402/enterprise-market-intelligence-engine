"""
config.py
---------
Centralised configuration for the Automated Enterprise Market Intelligence
& Fact-Checking Engine.

All settings are loaded from environment variables via pydantic-settings so
the application is 12-factor compliant with zero hard-coded secrets.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from typing import Optional

from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Logging bootstrap – every module imports the logger from here so formatting
# is consistent across the entire codebase.
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("intelligence_engine")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Required env vars
    -----------------
    OPENAI_API_KEY      – OpenAI API key
    TAVILY_API_KEY      – Tavily web-search API key

    Optional env vars
    -----------------
    QDRANT_URL          – Qdrant vector database URL  (default: http://localhost:6333)
    QDRANT_API_KEY      – Qdrant cloud API key        (default: None → anonymous)
    QDRANT_COLLECTION   – Collection name              (default: enterprise_intel)
    LLM_MODEL           – OpenAI model identifier      (default: gpt-4.1-mini)
    LLM_TEMPERATURE     – Sampling temperature         (default: 0)
    MAX_SEARCH_RESULTS  – Tavily results per query     (default: 5)
    MAX_VECTOR_RESULTS  – Qdrant hits per query        (default: 5)
    CIRCUIT_BREAKER_MAX – Max critic→generator loops   (default: 3)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Required secrets -------------------------------------------------------
    openai_api_key: SecretStr = Field(..., description="OpenAI API key")
    tavily_api_key: SecretStr = Field(..., description="Tavily web-search API key")

    # --- Optional Qdrant config -------------------------------------------------
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant vector database base URL",
    )
    qdrant_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Qdrant cloud API key (leave unset for local/anonymous)",
    )
    qdrant_collection: str = Field(
        default="enterprise_intel",
        description="Qdrant collection that holds enterprise knowledge chunks",
    )

    # --- LLM config -------------------------------------------------------------
    llm_model: str = Field(
        default="gpt-4.1-mini",
        description="OpenAI model identifier",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (0 = fully deterministic)",
    )

    # --- Retrieval config -------------------------------------------------------
    max_search_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum Tavily snippets to retrieve per query",
    )
    max_vector_results: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum Qdrant hits to retrieve per query",
    )

    # --- Circuit breaker --------------------------------------------------------
    circuit_breaker_max: int = Field(
        default=3,
        ge=1,
        description="Maximum number of critic→generator correction loops",
    )

    @field_validator("llm_temperature", mode="before")
    @classmethod
    def clamp_temperature(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


# ---------------------------------------------------------------------------
# Singleton helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, validated Settings instance."""
    settings = Settings()  # type: ignore[call-arg]  # pydantic-settings reads env
    logger.info(
        "Settings loaded — model=%s  qdrant=%s  collection=%s  circuit_breaker=%d",
        settings.llm_model,
        settings.qdrant_url,
        settings.qdrant_collection,
        settings.circuit_breaker_max,
    )
    return settings


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """
    Return a cached ChatOpenAI instance.

    Configured with temperature=0 for deterministic, auditable outputs and
    max_tokens=4096 to comfortably fit a structured JSON report plus any
    chain-of-thought preamble within a single completion.
    """
    cfg = get_settings()
    llm = ChatOpenAI(
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
        max_tokens=4096,
        openai_api_key=cfg.openai_api_key.get_secret_value(),
    )
    logger.info("ChatOpenAI client initialised — model=%s", cfg.llm_model)
    return llm
