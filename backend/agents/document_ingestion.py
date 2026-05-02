from io import BytesIO
from pathlib import Path
from typing import Dict, List

from PyPDF2 import PdfReader
from docx import Document as DocxDocument


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
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(page.strip() for page in pages if page and page.strip())


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


def chunk_document_text(text: str, source_name: str, chunk_size: int = 900, overlap: int = 150) -> List[Dict]:
    normalized_text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized_text:
        return []

    paragraphs = normalized_text.split("\n")
    chunks: List[Dict] = []
    current_parts: List[str] = []
    current_length = 0
    chunk_index = 1

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if current_parts and current_length + len(paragraph) > chunk_size:
            chunk_text = "\n".join(current_parts).strip()
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        "source": source_name,
                        "source_path": source_name,
                        "chunk_index": chunk_index,
                    },
                }
            )
            chunk_index += 1

            overlap_text = chunk_text[-overlap:] if overlap > 0 else ""
            current_parts = [overlap_text, paragraph] if overlap_text else [paragraph]
            current_length = sum(len(part) for part in current_parts)
            continue

        current_parts.append(paragraph)
        current_length += len(paragraph)

    if current_parts:
        chunks.append(
            {
                "content": "\n".join(current_parts).strip(),
                "metadata": {
                    "source": source_name,
                    "source_path": source_name,
                    "chunk_index": chunk_index,
                },
            }
        )

    return chunks
