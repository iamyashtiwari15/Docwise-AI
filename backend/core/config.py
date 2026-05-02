import os
from dataclasses import dataclass
from functools import lru_cache


def _split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    api_prefix: str
    cors_origins: list[str]
    llm_model_name: str
    embedding_model_name: str
    session_max_messages: int
    document_relevance_threshold: float
    web_search_max_results: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        api_prefix=os.getenv("API_PREFIX", "/api"),
        cors_origins=_split_csv(
            os.getenv("CORS_ORIGINS"),
            [
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:4173",
                "http://127.0.0.1:4173",
            ],
        ),
        llm_model_name=os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        session_max_messages=int(os.getenv("SESSION_MAX_MESSAGES", "30")),
        document_relevance_threshold=float(os.getenv("DOCUMENT_RELEVANCE_THRESHOLD", "0.35")),
        web_search_max_results=int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3")),
    )
