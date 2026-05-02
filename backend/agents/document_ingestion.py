from io import BytesIO
from pathlib import Path
import re
from typing import Dict, List

from pypdf import PdfReader
from docx import Document as DocxDocument

from core.config import get_settings


SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", ".docx", ".txt"}
SUPPORTED_DOCUMENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
}


def is_document_upload(filename: str | None, content_type: str | None) -> bool:
    extension = Path(filename or "").suffix.lower()
    normalized_type = (content_type or "").lower()
    return extension in SUPPORTED_DOCUMENT_EXTENSIONS or normalized_type in SUPPORTED_DOCUMENT_TYPES


def _extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            # Prefix each page with a marker so downstream chunking can record page numbers
            pages.append(f"[PAGE {i + 1}]\n{text.strip()}")
    return "\n\n".join(pages)


def _extract_docx_text(file_bytes: bytes) -> str:
    document = DocxDocument(BytesIO(file_bytes))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    return "\n".join(paragraphs)


def _extract_txt_text(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore").strip()


def extract_text_from_upload(filename: str, file_bytes: bytes, content_type: str | None = None) -> str:
    extension = Path(filename).suffix.lower()

    if extension == ".pdf" or (content_type or "").lower() == "application/pdf":
        return _extract_pdf_text(file_bytes)
    if extension == ".docx" or (content_type or "").lower() == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return _extract_docx_text(file_bytes)
    if extension == ".txt" or (content_type or "").lower() == "text/plain":
        return _extract_txt_text(file_bytes)

    raise ValueError("Unsupported document type. Please upload a PDF, DOCX, or TXT file.")


def _normalize_document_text(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        cleaned_line = re.sub(r"\s+", " ", raw_line).strip()
        lines.append(cleaned_line)

    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _split_long_unit(unit: str, max_length: int) -> List[str]:
    stripped = re.sub(r"\s+", " ", unit).strip()
    if not stripped:
        return []
    if len(stripped) <= max_length:
        return [stripped]

    sentences = re.split(r"(?<=[.!?])\s+", stripped)
    if len(sentences) <= 1:
        words = stripped.split()
        pieces: List[str] = []
        current_words: List[str] = []
        current_length = 0

        for word in words:
            projected = current_length + len(word) + (1 if current_words else 0)
            if current_words and projected > max_length:
                pieces.append(" ".join(current_words))
                current_words = [word]
                current_length = len(word)
                continue

            current_words.append(word)
            current_length = projected

        if current_words:
            pieces.append(" ".join(current_words))
        return pieces

    pieces: List[str] = []
    current_parts: List[str] = []
    current_length = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_length:
            if current_parts:
                pieces.append(" ".join(current_parts).strip())
                current_parts = []
                current_length = 0
            pieces.extend(_split_long_unit(sentence, max_length))
            continue

        projected = current_length + len(sentence) + (1 if current_parts else 0)
        if current_parts and projected > max_length:
            pieces.append(" ".join(current_parts).strip())
            current_parts = [sentence]
            current_length = len(sentence)
            continue

        current_parts.append(sentence)
        current_length = projected

    if current_parts:
        pieces.append(" ".join(current_parts).strip())

    return pieces


def _build_overlap_units(units: List[str], overlap: int) -> List[str]:
    if overlap <= 0 or not units:
        return []

    retained: List[str] = []
    retained_length = 0
    for unit in reversed(units):
        additional_length = len(unit) + (1 if retained else 0)
        if retained and retained_length + additional_length > overlap:
            break
        retained.insert(0, unit)
        retained_length += additional_length

    return retained


def chunk_document_text(text: str, source_name: str, chunk_size: int | None = None, overlap: int | None = None) -> List[Dict]:
    settings = get_settings()
    resolved_chunk_size = chunk_size or settings.document_chunk_size
    resolved_overlap = overlap if overlap is not None else settings.document_chunk_overlap

    normalized_text = _normalize_document_text(text)
    if not normalized_text:
        return []

    paragraph_blocks = [block.strip() for block in re.split(r"\n\s*\n", normalized_text) if block.strip()]
    if not paragraph_blocks:
        paragraph_blocks = [line.strip() for line in normalized_text.splitlines() if line.strip()]

    units: List[str] = []
    for block in paragraph_blocks:
        units.extend(_split_long_unit(block, resolved_chunk_size))

    if not units:
        return []

    chunks: List[Dict] = []
    current_units: List[str] = []
    current_length = 0
    chunk_index = 1

    for unit in units:
        projected = current_length + len(unit) + (1 if current_units else 0)
        if current_units and projected > resolved_chunk_size:
            chunk_text = "\n".join(current_units).strip()
            metadata: Dict = {
                "source": source_name,
                "source_path": source_name,
                "chunk_index": chunk_index,
            }
            page_match = re.search(r"\[PAGE (\d+)\]", chunk_text)
            if page_match:
                metadata["page_number"] = int(page_match.group(1))
            chunks.append({"content": chunk_text, "metadata": metadata})
            chunk_index += 1

            overlap_units = _build_overlap_units(current_units, resolved_overlap)
            current_units = list(overlap_units)
            current_length = sum(len(part) for part in current_units) + max(len(current_units) - 1, 0)

        projected = current_length + len(unit) + (1 if current_units else 0)
        if current_units and projected > resolved_chunk_size:
            current_units = []
            current_length = 0

        current_units.append(unit)
        current_length += len(unit) + (1 if len(current_units) > 1 else 0)

    if current_units:
        chunk_text = "\n".join(current_units).strip()
        metadata = {
            "source": source_name,
            "source_path": source_name,
            "chunk_index": chunk_index,
        }
        page_match = re.search(r"\[PAGE (\d+)\]", chunk_text)
        if page_match:
            metadata["page_number"] = int(page_match.group(1))
        chunks.append({"content": chunk_text, "metadata": metadata})

    return chunks
