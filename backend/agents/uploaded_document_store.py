from dataclasses import dataclass
from threading import Lock
from typing import Dict, List

from agents.document_ingestion import chunk_document_text, extract_text_from_upload
from agents.llm_loader import get_embedding_model


@dataclass
class StoredChunk:
    content: str
    metadata: Dict
    embedding: List[float]


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
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
        self._lock = Lock()
        self._session_chunks: Dict[str, List[StoredChunk]] = {}
        self._session_files: Dict[str, List[str]] = {}

    def _get_embedding_model(self):
        if self.embedding_model is None:
            self.embedding_model = get_embedding_model()
        return self.embedding_model

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

    def retrieve(self, session_id: str, query: str, top_k: int = 4) -> List[Dict]:
        with self._lock:
            chunks = list(self._session_chunks.get(session_id, []))

        if not chunks:
            return []

        query_embedding = [float(value) for value in self._get_embedding_model().embed_query(query)]
        scored_chunks = []
        for chunk in chunks:
            score = _cosine_similarity(query_embedding, chunk.embedding)
            scored_chunks.append(
                {
                    "content": chunk.content,
                    "score": score,
                    "source": chunk.metadata.get("source", "uploaded_document"),
                    "source_path": chunk.metadata.get("source_path", chunk.metadata.get("source", "uploaded_document")),
                    "metadata": {
                        **chunk.metadata,
                        "score": score,
                    },
                }
            )

        scored_chunks.sort(key=lambda item: item["score"], reverse=True)
        return scored_chunks[:top_k]


uploaded_document_store = UploadedDocumentStore()
