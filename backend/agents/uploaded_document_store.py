from dataclasses import dataclass
import logging
from threading import Lock
from typing import Dict, List

from agents.document_ingestion import chunk_document_text, extract_text_from_upload
from agents.llm_loader import get_embedding_model
from agents.retrieval_utils import (
    extract_query_terms,
    generate_query_variants,
    lexical_overlap_score,
    should_use_llm_multi_query,
)
from agents.rag_agent.role_llm_loader import get_llm as get_role_llm
from core.config import get_settings


logger = logging.getLogger(__name__)


@dataclass
class StoredChunk:
    content: str
    metadata: Dict
    embedding: List[float]
    terms: frozenset[str]


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if left is None or right is None:
        return 0.0

    left_values = list(left)
    right_values = list(right)
    if not left_values or not right_values or len(left_values) != len(right_values):
        return 0.0

    dot_product = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = sum(value * value for value in left_values) ** 0.5
    right_norm = sum(value * value for value in right_values) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


class UploadedDocumentStore:
    def __init__(self, embedding_model=None, query_variant_llm=None):
        self.embedding_model = embedding_model
        self.query_variant_llm = query_variant_llm
        self._lock = Lock()
        self._session_chunks: Dict[str, List[StoredChunk]] = {}
        self._session_files: Dict[str, List[str]] = {}

    def _get_embedding_model(self):
        if self.embedding_model is None:
            self.embedding_model = get_embedding_model()
        return self.embedding_model

    def _get_query_variant_llm(self):
        if self.query_variant_llm is None:
            self.query_variant_llm = get_role_llm(role="multi_query")
        return self.query_variant_llm

    def add_file(self, session_id: str, filename: str, file_bytes: bytes, content_type: str | None = None) -> Dict:
        text = extract_text_from_upload(filename, file_bytes, content_type=content_type)
        if not text.strip():
            raise ValueError("The uploaded file does not contain readable text.")

        chunks = chunk_document_text(text, filename)
        if not chunks:
            raise ValueError("The uploaded file does not contain enough text to index.")

        embeddings = self._get_embedding_model().embed_documents([chunk["content"] for chunk in chunks])
        stored_chunks = [
            StoredChunk(
                content=chunk["content"],
                metadata=chunk["metadata"],
                embedding=[float(value) for value in embedding],
                terms=frozenset(extract_query_terms(chunk["content"])),
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        with self._lock:
            self._session_chunks.setdefault(session_id, []).extend(stored_chunks)
            self._session_files.setdefault(session_id, []).append(filename)

        return {
            "filename": filename,
            "chunk_count": len(stored_chunks),
            "file_count": len(self._session_files.get(session_id, [])),
        }

    def has_documents(self, session_id: str) -> bool:
        with self._lock:
            return bool(self._session_chunks.get(session_id))

    def list_files(self, session_id: str) -> List[str]:
        with self._lock:
            return list(self._session_files.get(session_id, []))

    def retrieve(self, session_id: str, query: str, top_k: int | None = None, candidate_k: int | None = None) -> List[Dict]:
        settings = get_settings()
        with self._lock:
            chunks = list(self._session_chunks.get(session_id, []))

        if not chunks:
            return []

        resolved_top_k = top_k or settings.rag_top_k
        resolved_candidate_k = max(candidate_k or settings.rag_candidate_k, resolved_top_k)
        query_variants = [query]
        if settings.enable_multi_query_retrieval:
            query_variant_llm = None
            if settings.enable_llm_multi_query and should_use_llm_multi_query(query, settings.llm_multi_query_min_terms):
                try:
                    query_variant_llm = self._get_query_variant_llm()
                except Exception as error:
                    logger.warning("[RETRIEVE] LLM multi-query unavailable, falling back to heuristics: %s", error)

            query_variants = generate_query_variants(query, settings.rag_max_query_variants, llm=query_variant_llm)

        logger.info(
            "[RETRIEVE] session=%s | chunks=%d | query_variants=%d | top_k=%d | candidate_k=%d",
            session_id,
            len(chunks),
            len(query_variants),
            resolved_top_k,
            resolved_candidate_k,
        )

        aggregated_candidates: Dict[tuple[str, int, str], Dict] = {}
        embedding_model = self._get_embedding_model()

        for variant in query_variants:
            variant_embedding = [float(value) for value in embedding_model.embed_query(variant)]
            variant_terms = extract_query_terms(variant)
            scored_chunks = []
            for chunk in chunks:
                semantic_score = _cosine_similarity(variant_embedding, chunk.embedding)
                lexical_score = lexical_overlap_score(variant_terms, chunk.terms)
                retrieval_score = semantic_score * 0.9 + lexical_score * 0.1
                scored_chunks.append((chunk, semantic_score, lexical_score, retrieval_score))

            scored_chunks.sort(key=lambda item: item[3], reverse=True)
            for rank, (chunk, semantic_score, lexical_score, retrieval_score) in enumerate(
                scored_chunks[:resolved_candidate_k],
                start=1,
            ):
                source = chunk.metadata.get("source", "uploaded_document")
                source_path = chunk.metadata.get("source_path", source)
                chunk_index = int(chunk.metadata.get("chunk_index", rank))
                candidate_key = (source_path, chunk_index, chunk.content)

                candidate = aggregated_candidates.setdefault(
                    candidate_key,
                    {
                        "content": chunk.content,
                        "source": source,
                        "source_path": source_path,
                        "metadata": dict(chunk.metadata),
                        "score": 0.0,
                        "rerank_score": 0.0,
                        "combined_score": 0.0,
                        "variant_hits": 0,
                        "best_rank": rank,
                        "lexical_score": 0.0,
                        "retrieval_score": 0.0,
                    },
                )
                candidate["score"] = max(candidate["score"], semantic_score)
                candidate["lexical_score"] = max(candidate["lexical_score"], lexical_score)
                candidate["retrieval_score"] = max(candidate["retrieval_score"], retrieval_score)
                candidate["variant_hits"] += 1
                candidate["best_rank"] = min(candidate["best_rank"], rank)

        total_variants = max(len(query_variants), 1)
        results = []
        for candidate in aggregated_candidates.values():
            multi_query_support = candidate["variant_hits"] / total_variants if total_variants > 1 else 0.0
            rerank_score = candidate["retrieval_score"]
            combined_score = (
                rerank_score * 0.95 + multi_query_support * 0.05
                if settings.enable_lightweight_rerank
                else candidate["score"]
            )

            metadata = {
                **candidate["metadata"],
                "score": candidate["score"],
                "rerank_score": rerank_score,
                "combined_score": combined_score,
                "variant_hits": candidate["variant_hits"],
                "best_rank": candidate["best_rank"],
            }
            results.append(
                {
                    "content": candidate["content"],
                    "score": candidate["score"],
                    "rerank_score": rerank_score,
                    "combined_score": combined_score,
                    "source": candidate["source"],
                    "source_path": candidate["source_path"],
                    "metadata": metadata,
                }
            )

        results.sort(
            key=lambda item: (
                item.get("combined_score", item.get("rerank_score", item.get("score", 0.0))),
                item.get("score", 0.0),
                -item.get("metadata", {}).get("best_rank", 9999),
            ),
            reverse=True,
        )
        return results[:resolved_top_k]


uploaded_document_store = UploadedDocumentStore()
