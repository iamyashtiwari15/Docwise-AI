import os
from dataclasses import dataclass
from functools import lru_cache


def _split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    api_prefix: str
    cors_origins: list[str]
    llm_model_name: str
    embedding_model_name: str
    embedding_device: str
    session_max_messages: int
    document_relevance_threshold: float
    web_search_max_results: int
    log_level: str
    rag_top_k: int
    rag_candidate_k: int
    rag_max_query_variants: int
    document_chunk_size: int
    document_chunk_overlap: int
    enable_multi_query_retrieval: bool
    enable_llm_multi_query: bool
    enable_lightweight_rerank: bool
    llm_multi_query_min_terms: int
    # accuracy-first additions
    enable_cross_encoder_rerank: bool
    cross_encoder_model_name: str
    enable_hyde: bool
    rag_window_size: int


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
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5"),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        session_max_messages=int(os.getenv("SESSION_MAX_MESSAGES", "30")),
        document_relevance_threshold=float(os.getenv("DOCUMENT_RELEVANCE_THRESHOLD", "0.30")),
        web_search_max_results=int(os.getenv("WEB_SEARCH_MAX_RESULTS", "3")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        rag_top_k=int(os.getenv("RAG_TOP_K", "6")),
        rag_candidate_k=int(os.getenv("RAG_CANDIDATE_K", "20")),
        rag_max_query_variants=int(os.getenv("RAG_MAX_QUERY_VARIANTS", "5")),
        document_chunk_size=int(os.getenv("DOCUMENT_CHUNK_SIZE", "512")),
        document_chunk_overlap=int(os.getenv("DOCUMENT_CHUNK_OVERLAP", "100")),
        enable_multi_query_retrieval=_env_flag("ENABLE_MULTI_QUERY_RETRIEVAL", True),
        enable_llm_multi_query=_env_flag("ENABLE_LLM_MULTI_QUERY", True),
        enable_lightweight_rerank=_env_flag("ENABLE_LIGHTWEIGHT_RERANK", True),
        llm_multi_query_min_terms=int(os.getenv("LLM_MULTI_QUERY_MIN_TERMS", "5")),
        enable_cross_encoder_rerank=_env_flag("ENABLE_CROSS_ENCODER_RERANK", True),
        cross_encoder_model_name=os.getenv("CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-12-v2"),
        enable_hyde=_env_flag("ENABLE_HYDE", True),
        rag_window_size=int(os.getenv("RAG_WINDOW_SIZE", "1")),
    )
