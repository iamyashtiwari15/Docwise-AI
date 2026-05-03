import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env path relative to this file (backend/core/ → project root)
_ENV_FILE = Path(__file__).parent.parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API
    api_prefix: str = "/api"
    cors_origins: List[str] = Field(
        default=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ]
    )

    # LLM
    llm_model_name: str = "llama-3.3-70b-versatile"

    # Embedding
    embedding_model_name: str = "microsoft/harrier-oss-v1-270m"
    embedding_device: str = "cpu"

    # Session
    session_max_messages: int = 30

    # RAG retrieval
    document_relevance_threshold: float = 0.20
    rag_top_k: int = 6
    rag_candidate_k: int = 50
    rag_max_query_variants: int = 5
    rag_window_size: int = 2

    # Chunking
    document_chunk_size: int = 1200
    document_chunk_overlap: int = 300

    # Feature flags
    enable_multi_query_retrieval: bool = True
    enable_llm_multi_query: bool = True
    enable_lightweight_rerank: bool = True
    enable_cross_encoder_rerank: bool = True
    enable_hyde: bool = True

    # Reranker
    cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    llm_multi_query_min_terms: int = 5

    # Vision (for describing embedded images in uploaded documents)
    vision_model_name: str = "llama-3.2-11b-vision-preview"

    # Web search
    web_search_max_results: int = 3

    # Logging
    log_level: str = "INFO"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Accept comma-separated string from env or a list."""
        if isinstance(v, str):
            return [item.strip() for item in v.split(",") if item.strip()]
        return v

    @field_validator("embedding_device", mode="before")
    @classmethod
    def validate_device(cls, v):
        allowed = {"cpu", "cuda", "mps"}
        v = str(v).strip().lower()
        if v not in allowed:
            raise ValueError(f"embedding_device must be one of {allowed}, got '{v}'")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
