import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from threading import Event, Lock
from typing import Dict, List, Optional, Tuple

from agents.document_ingestion import chunk_document_text, extract_text_from_upload, save_chunks_debug
from agents.llm_loader import get_embedding_model
from agents.retrieval_utils import (
    extract_query_terms,
    generate_query_variants,
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
    terms: frozenset


# ── Helpers ──────────────────────────────────────────────────────────────────

def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_l = sum(v * v for v in left) ** 0.5
    norm_r = sum(v * v for v in right) ** 0.5
    if norm_l == 0 or norm_r == 0:
        return 0.0
    return dot / (norm_l * norm_r)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))


def _rrf_score(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score for a given 0-based rank."""
    return 1.0 / (k + rank + 1)


@lru_cache(maxsize=1)
def _get_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder
    logger.info("[RERANK] Loading cross-encoder model: %s", model_name)
    return CrossEncoder(model_name)


# ── Ingestion status tracking ─────────────────────────────────────────────────

class IngestionStatus(str, Enum):
    PENDING  = "pending"
    DONE     = "done"
    ERROR    = "error"


@dataclass
class _IngestionEntry:
    status: IngestionStatus = IngestionStatus.PENDING
    event: Event = field(default_factory=Event)
    error: Optional[str] = None
    chunk_count: int = 0


class IngestionTracker:
    """
    Tracks per-session per-file ingestion state.
    Callers can wait() until a file is done indexing before querying.
    """
    def __init__(self):
        self._lock = Lock()
        self._entries: Dict[str, _IngestionEntry] = {}  # key = "session_id::filename"

    @staticmethod
    def _key(session_id: str, filename: str) -> str:
        return f"{session_id}::{filename}"

    def start(self, session_id: str, filename: str) -> bool:
        """Register a file as PENDING. Returns False if already tracked (skip re-ingest)."""
        key = self._key(session_id, filename)
        with self._lock:
            if key in self._entries:
                return False
            self._entries[key] = _IngestionEntry()
            return True

    def finish(self, session_id: str, filename: str, chunk_count: int) -> None:
        key = self._key(session_id, filename)
        with self._lock:
            entry = self._entries.get(key)
        if entry:
            entry.chunk_count = chunk_count
            entry.status = IngestionStatus.DONE
            entry.event.set()

    def fail(self, session_id: str, filename: str, error: str) -> None:
        key = self._key(session_id, filename)
        with self._lock:
            entry = self._entries.get(key)
        if entry:
            entry.error = error
            entry.status = IngestionStatus.ERROR
            entry.event.set()

    def wait(self, session_id: str, filename: str, timeout: float = 120.0) -> Optional["_IngestionEntry"]:
        """Block until ingestion completes (or timeout). Returns the entry."""
        key = self._key(session_id, filename)
        with self._lock:
            entry = self._entries.get(key)
        if not entry:
            return None
        entry.event.wait(timeout=timeout)
        return entry

    def is_done(self, session_id: str, filename: str) -> bool:
        key = self._key(session_id, filename)
        with self._lock:
            entry = self._entries.get(key)
        return entry is not None and entry.status == IngestionStatus.DONE

    def get_status(self, session_id: str, filename: str) -> Optional[IngestionStatus]:
        key = self._key(session_id, filename)
        with self._lock:
            entry = self._entries.get(key)
        return entry.status if entry else None


ingestion_tracker = IngestionTracker()


# ── Store ─────────────────────────────────────────────────────────────────────

class UploadedDocumentStore:
    def __init__(self, embedding_model=None, query_variant_llm=None):
        self.embedding_model = embedding_model
        self.query_variant_llm = query_variant_llm
        self._lock = Lock()
        self._session_chunks: Dict[str, List[StoredChunk]] = {}
        self._session_files: Dict[str, List[str]] = {}
        # BM25 index cache (rebuilt lazily when new docs are added)
        self._bm25_cache: Dict[str, object] = {}
        self._bm25_dirty: Dict[str, bool] = {}

    def _get_embedding_model(self):
        if self.embedding_model is None:
            self.embedding_model = get_embedding_model()
        return self.embedding_model

    def _get_query_variant_llm(self):
        if self.query_variant_llm is None:
            self.query_variant_llm = get_role_llm(role="multi_query")
        return self.query_variant_llm

    def _get_hyde_llm(self):
        return get_role_llm(role="hyde")

    # ── BM25 ──────────────────────────────────────────────────────────────────

    def _get_bm25(self, session_id: str, chunks: List[StoredChunk]):
        if self._bm25_dirty.get(session_id, True) or session_id not in self._bm25_cache:
            try:
                from rank_bm25 import BM25Okapi
                tokenized = [list(c.terms) for c in chunks]
                self._bm25_cache[session_id] = BM25Okapi(tokenized) if tokenized else None
            except ImportError:
                logger.warning("[BM25] rank_bm25 not installed — skipping BM25 retrieval")
                self._bm25_cache[session_id] = None
            self._bm25_dirty[session_id] = False
        return self._bm25_cache.get(session_id)

    # ── HyDE ──────────────────────────────────────────────────────────────────

    def _generate_hyde_embedding(self, query: str) -> Optional[List[float]]:
        """Embed a hypothetical answer to improve dense retrieval recall."""
        try:
            result = self._get_hyde_llm().invoke(query)
            hypothesis = (result.content if hasattr(result, "content") else str(result)).strip()
            if not hypothesis:
                return None
            logger.info("[HyDE] Generated hypothetical passage (%d chars)", len(hypothesis))
            return [float(v) for v in self._get_embedding_model().embed_query(hypothesis)]
        except Exception as exc:
            logger.warning("[HyDE] Skipped (error): %s", exc)
            return None

    # ── Context window ────────────────────────────────────────────────────────

    def _expand_with_context_window(
        self,
        results: List[Dict],
        chunks: List[StoredChunk],
        window_size: int,
    ) -> List[Dict]:
        """Prepend/append neighboring chunks to each retrieved result."""
        if window_size <= 0:
            return results

        # Build lookup: (source_path, chunk_index) -> content
        lookup: Dict[Tuple[str, int], str] = {}
        for c in chunks:
            src = c.metadata.get("source_path", c.metadata.get("source", ""))
            idx = int(c.metadata.get("chunk_index", 0))
            lookup[(src, idx)] = c.content

        expanded = []
        for res in results:
            src = res.get("source_path", res.get("source", ""))
            ci = int(res.get("metadata", {}).get("chunk_index", 0))
            if ci == 0:
                expanded.append(res)
                continue

            before = [
                lookup[(src, ci - off)]
                for off in range(window_size, 0, -1)
                if (src, ci - off) in lookup
            ]
            after = [
                lookup[(src, ci + off)]
                for off in range(1, window_size + 1)
                if (src, ci + off) in lookup
            ]

            if not before and not after:
                expanded.append(res)
                continue

            pieces = before + [res["content"]] + after
            res = dict(res)
            res["content"] = "\n\n".join(pieces)
            expanded.append(res)

        return expanded

    # ── Public API ────────────────────────────────────────────────────────────

    def add_file(self, session_id: str, filename: str, file_bytes: bytes, content_type: str | None = None) -> Dict:
        text = extract_text_from_upload(filename, file_bytes, content_type=content_type)
        if not text.strip():
            raise ValueError("The uploaded file does not contain readable text.")

        chunks = chunk_document_text(text, filename)
        if not chunks:
            raise ValueError("The uploaded file does not contain enough text to index.")

        # Save human-readable chunk dump for quality inspection
        try:
            save_chunks_debug(chunks, filename)
        except Exception as _exc:
            logger.warning("[CHUNK_DEBUG] Could not save debug file: %s", _exc)

        embeddings= self._get_embedding_model().embed_documents([c["content"] for c in chunks])
        stored_chunks = [
            StoredChunk(
                content=chunk["content"],
                metadata=chunk["metadata"],
                embedding=[float(v) for v in emb],
                terms=frozenset(extract_query_terms(chunk["content"])),
            )
            for chunk, emb in zip(chunks, embeddings)
        ]

        with self._lock:
            self._session_chunks.setdefault(session_id, []).extend(stored_chunks)
            self._session_files.setdefault(session_id, []).append(filename)
            self._bm25_dirty[session_id] = True

        return {
            "filename": filename,
            "chunk_count": len(stored_chunks),
            "file_count": len(self._session_files.get(session_id, [])),
        }

    def has_documents(self, session_id: str) -> bool:
        with self._lock:
            return bool(self._session_chunks.get(session_id))

    def chunk_count(self, session_id: str) -> int:
        with self._lock:
            return len(self._session_chunks.get(session_id, []))

    def list_files(self, session_id: str) -> List[str]:
        with self._lock:
            return list(self._session_files.get(session_id, []))

    def retrieve(
        self,
        session_id: str,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
    ) -> List[Dict]:
        settings = get_settings()
        with self._lock:
            chunks = list(self._session_chunks.get(session_id, []))

        if not chunks:
            return []

        resolved_top_k = top_k or settings.rag_top_k
        # Use a generous candidate pool so cross-encoder has good material to rerank
        resolved_candidate_k = max(candidate_k or settings.rag_candidate_k, resolved_top_k * 4)

        # ── 1. Query variants ─────────────────────────────────────────────────
        query_variants = [query]
        if settings.enable_multi_query_retrieval:
            llm_for_variants = None
            if settings.enable_llm_multi_query and should_use_llm_multi_query(query, settings.llm_multi_query_min_terms):
                try:
                    llm_for_variants = self._get_query_variant_llm()
                except Exception as exc:
                    logger.warning("[RETRIEVE] LLM multi-query unavailable: %s", exc)
            query_variants = generate_query_variants(query, settings.rag_max_query_variants, llm=llm_for_variants)

        # ── 2. HyDE embedding ─────────────────────────────────────────────────
        hyde_embedding: Optional[List[float]] = None
        if settings.enable_hyde:
            hyde_embedding = self._generate_hyde_embedding(query)

        logger.info(
            "[RETRIEVE] session=%s | chunks=%d | variants=%d | hyde=%s | top_k=%d | candidate_k=%d",
            session_id, len(chunks), len(query_variants), hyde_embedding is not None,
            resolved_top_k, resolved_candidate_k,
        )

        # ── 3. Dense embeddings for all variants (+ HyDE) ─────────────────────
        embedding_model = self._get_embedding_model()
        all_query_embeddings: List[List[float]] = [
            [float(v) for v in embedding_model.embed_query(v)] for v in query_variants
        ]
        if hyde_embedding:
            all_query_embeddings.append(hyde_embedding)

        # ── 4. BM25 index ─────────────────────────────────────────────────────
        bm25 = self._get_bm25(session_id, chunks)

        # ── 5. Hybrid retrieval → RRF fusion ──────────────────────────────────
        rrf_scores: Dict[int, float] = {}

        # Dense: one ranking per query embedding
        for emb in all_query_embeddings:
            dense_scores = [_cosine_similarity(emb, c.embedding) for c in chunks]
            dense_ranking = sorted(range(len(chunks)), key=lambda i: dense_scores[i], reverse=True)
            for rank, idx in enumerate(dense_ranking[:resolved_candidate_k]):
                rrf_scores[idx] = rrf_scores.get(idx, 0.0) + _rrf_score(rank)

        # BM25: one ranking per query variant
        if bm25 is not None:
            for variant in query_variants:
                terms = list(extract_query_terms(variant))
                if not terms:
                    continue
                bm25_raw = bm25.get_scores(terms)
                bm25_ranking = sorted(range(len(chunks)), key=lambda i: bm25_raw[i], reverse=True)
                for rank, idx in enumerate(bm25_ranking[:resolved_candidate_k]):
                    rrf_scores[idx] = rrf_scores.get(idx, 0.0) + _rrf_score(rank)

        # Sort candidates by combined RRF score
        sorted_candidates = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)[:resolved_candidate_k]

        # ── 6. Cross-encoder reranking ────────────────────────────────────────
        ce_score_map: Dict[int, float] = {}
        if settings.enable_cross_encoder_rerank and sorted_candidates:
            try:
                cross_encoder = _get_cross_encoder(settings.cross_encoder_model_name)
                pairs = [(query, chunks[i].content) for i in sorted_candidates]
                logits = cross_encoder.predict(pairs)
                ce_scores = [_sigmoid(float(s)) for s in logits]
                sorted_candidates = [
                    sorted_candidates[j]
                    for j in sorted(range(len(sorted_candidates)), key=lambda j: ce_scores[j], reverse=True)
                ]
                ce_score_map = {sorted_candidates[j]: ce_scores[j] for j in range(len(sorted_candidates))}
                logger.info("[RERANK] Cross-encoder applied | top_score=%.3f", max(ce_scores))
            except Exception as exc:
                logger.warning("[RERANK] Cross-encoder failed, using RRF order: %s", exc)

        # ── 7. Build result dicts for top-k ──────────────────────────────────
        results: List[Dict] = []
        for idx in sorted_candidates[:resolved_top_k]:
            chunk = chunks[idx]
            rrf = rrf_scores.get(idx, 0.0)
            ce = ce_score_map.get(idx)

            best_semantic = max(
                _cosine_similarity(emb, chunk.embedding) for emb in all_query_embeddings
            )
            combined = ce if ce is not None else rrf

            results.append({
                "content": chunk.content,
                "score": best_semantic,
                "rerank_score": combined,
                "combined_score": combined,
                "source": chunk.metadata.get("source", "uploaded_document"),
                "source_path": chunk.metadata.get("source_path", chunk.metadata.get("source", "")),
                "metadata": {
                    **chunk.metadata,
                    "score": best_semantic,
                    "rerank_score": combined,
                    "combined_score": combined,
                    "rrf_score": rrf,
                },
            })

        # ── 8. Context window expansion ───────────────────────────────────────
        if settings.rag_window_size > 0:
            results = self._expand_with_context_window(results, chunks, settings.rag_window_size)

        logger.info(
            "[RETRIEVE] Done | returned=%d | top_score=%.3f",
            len(results),
            results[0]["combined_score"] if results else 0.0,
        )
        return results


uploaded_document_store = UploadedDocumentStore()
