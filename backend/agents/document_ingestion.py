import base64
import logging
import os
from io import BytesIO
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF — installed as part of pymupdf4llm
import pymupdf4llm
from docx import Document as DocxDocument

from core.config import get_settings

logger = logging.getLogger(__name__)


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


# ── Image helpers ──────────────────────────────────────────────────────────────

def _extract_page_images(doc: fitz.Document, page_index: int) -> List[Tuple[bytes, str]]:
    """
    Extract content images from a PDF page.
    Filters out tiny decorative images (icons, borders, logos).
    Returns list of (image_bytes, extension) tuples.
    """
    MIN_BYTES = 4096    # skip images < 4 KB
    MIN_DIMENSION = 60  # skip images narrower/shorter than 60 px

    page = doc[page_index]
    results: List[Tuple[bytes, str]] = []
    seen_xrefs: set = set()

    for img_info in page.get_images(full=True):
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            base_img = doc.extract_image(xref)
            img_bytes: bytes = base_img["image"]
            img_ext: str = base_img.get("ext", "jpeg")
            width: int = base_img.get("width", 0)
            height: int = base_img.get("height", 0)
            if len(img_bytes) < MIN_BYTES or width < MIN_DIMENSION or height < MIN_DIMENSION:
                continue
            results.append((img_bytes, img_ext))
        except Exception:
            continue
    return results


def _describe_image_bytes(img_bytes: bytes, img_ext: str) -> Optional[str]:
    """
    Send an image to the Groq vision LLM and return a structured textual description.
    Used to produce [IMAGE SUMMARY: ...] markers embedded in the document text so the
    RAG pipeline can answer questions about charts, diagrams, and images.
    Returns None on failure so callers can safely skip.
    """
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage

        settings = get_settings()
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return None

        mime_map = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "bmp": "image/bmp", "webp": "image/webp",
        }
        mime = mime_map.get(img_ext.lower().lstrip("."), "image/jpeg")
        b64 = base64.b64encode(img_bytes).decode()

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model=settings.vision_model_name,
            temperature=0,
        )
        msg = HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            },
            {
                "type": "text",
                "text": (
                    "Describe this image concisely but completely for document Q&A retrieval. "
                    "If it contains a chart or graph: state the type, axes labels, data values, and key trends. "
                    "If it contains a diagram or flowchart: describe the structure and relationships shown. "
                    "If it is a table rendered as an image: transcribe all cell values row by row. "
                    "If it is a photo or illustration: describe exactly what is depicted. "
                    "Begin your description directly without 'This image shows' or similar preamble."
                ),
            },
        ])
        result = llm.invoke([msg])
        description = (result.content or "").strip()
        return description if description else None

    except Exception as exc:
        logger.warning("[IMAGE] Failed to describe embedded image (%s): %s", img_ext, exc)
        return None


def _extract_pdf_text(file_bytes: bytes) -> str:
    """
    Extract text from a PDF using pymupdf4llm (LLM-ready markdown per page).
    For each page:
      - Text content is preserved with headings, lists, and pipe-table formatting.
      - Embedded images are extracted, described via the Groq vision LLM, and
        injected as [IMAGE SUMMARY: ...] blocks at the end of the page content.
    Each page is prefixed with [PAGE X] so downstream chunking can record page numbers.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page_md_chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)

    pages: List[str] = []
    for md_chunk in page_md_chunks:
        page_text = (md_chunk.get("text") or "").strip()
        page_num = md_chunk.get("metadata", {}).get("page", 0) + 1

        parts: List[str] = [f"[PAGE {page_num}]"]
        if page_text:
            parts.append(page_text)

        # Describe embedded images on this page and inject summaries
        for img_bytes, img_ext in _extract_page_images(doc, page_num - 1):
            logger.info("[IMAGE] Describing embedded image on page %d (%s, %d bytes)", page_num, img_ext, len(img_bytes))
            description = _describe_image_bytes(img_bytes, img_ext)
            if description:
                parts.append(f"[IMAGE SUMMARY: {description}]")

        if len(parts) > 1:  # at least page marker + some content
            pages.append("\n\n".join(parts))

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
        # Preserve table rows and special markers — only collapse internal whitespace
        stripped = raw_line.strip()
        if stripped.startswith("|") or stripped.startswith("[PAGE ") or stripped.startswith("[IMAGE SUMMARY:"):
            lines.append(stripped)
        else:
            lines.append(re.sub(r"\s+", " ", raw_line).strip())

    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _is_markdown_table(block: str) -> bool:
    """
    Returns True if a paragraph block is a markdown pipe-table.
    A block qualifies if at least 60% of its non-empty lines start with '|'.
    """
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    table_lines = sum(1 for ln in lines if ln.startswith("|"))
    return (table_lines / len(lines)) >= 0.6


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


def _flush_chunk(units: List[str], source_name: str, chunk_index: int, extra_meta: Optional[Dict] = None) -> Dict:
    chunk_text = "\n".join(units).strip()
    metadata: Dict = {
        "source": source_name,
        "source_path": source_name,
        "chunk_index": chunk_index,
    }
    if extra_meta:
        metadata.update(extra_meta)
    page_match = re.search(r"\[PAGE (\d+)\]", chunk_text)
    if page_match:
        metadata["page_number"] = int(page_match.group(1))
    return {"content": chunk_text, "metadata": metadata}


def chunk_document_text(text: str, source_name: str, chunk_size: int | None = None, overlap: int | None = None) -> List[Dict]:
    """
    Split document text into overlapping chunks for vector indexing.

    Atomic rules:
    - Markdown pipe-tables are NEVER split; they are always emitted as a
      single self-contained chunk with metadata["is_table"] = True.
    - [IMAGE SUMMARY: ...] blocks follow the same rule.
    - All other text follows the normal sentence/word splitting with overlap.
    """
    settings = get_settings()
    resolved_chunk_size = chunk_size or settings.document_chunk_size
    resolved_overlap = overlap if overlap is not None else settings.document_chunk_overlap

    normalized_text = _normalize_document_text(text)
    if not normalized_text:
        return []

    paragraph_blocks = [block.strip() for block in re.split(r"\n\s*\n", normalized_text) if block.strip()]
    if not paragraph_blocks:
        paragraph_blocks = [line.strip() for line in normalized_text.splitlines() if line.strip()]

    # Build (text, is_atomic) pairs
    # Tables and IMAGE SUMMARY blocks are atomic — never split, never interleaved with normal text
    typed_units: List[Tuple[str, bool]] = []
    for block in paragraph_blocks:
        is_image_summary = block.startswith("[IMAGE SUMMARY:")
        if _is_markdown_table(block) or is_image_summary:
            typed_units.append((block, True))
        else:
            for piece in _split_long_unit(block, resolved_chunk_size):
                typed_units.append((piece, False))

    if not typed_units:
        return []

    chunks: List[Dict] = []
    current_units: List[str] = []
    current_length = 0
    chunk_index = 1

    for unit_text, is_atomic in typed_units:
        if is_atomic:
            # Flush whatever we have been accumulating first
            if current_units:
                chunks.append(_flush_chunk(current_units, source_name, chunk_index))
                chunk_index += 1
                current_units = []
                current_length = 0
            # Emit atomic block as its own standalone chunk
            is_table = _is_markdown_table(unit_text)
            extra = {"is_table": True} if is_table else {"is_image_summary": True}
            chunks.append(_flush_chunk([unit_text], source_name, chunk_index, extra))
            chunk_index += 1
            continue

        # Normal unit: accumulate with size limit
        projected = current_length + len(unit_text) + (1 if current_units else 0)
        if current_units and projected > resolved_chunk_size:
            chunks.append(_flush_chunk(current_units, source_name, chunk_index))
            chunk_index += 1

            overlap_units = _build_overlap_units(current_units, resolved_overlap)
            current_units = list(overlap_units)
            current_length = sum(len(p) for p in current_units) + max(len(current_units) - 1, 0)

        projected = current_length + len(unit_text) + (1 if current_units else 0)
        if current_units and projected > resolved_chunk_size:
            current_units = []
            current_length = 0

        current_units.append(unit_text)
        current_length += len(unit_text) + (1 if len(current_units) > 1 else 0)

    if current_units:
        chunks.append(_flush_chunk(current_units, source_name, chunk_index))

    return chunks


# ── Debug / quality-inspection helper ─────────────────────────────────────────

def save_chunks_debug(chunks: List[Dict], source_name: str) -> str:
    """
    Write all chunks with their metadata to a human-readable .txt file inside
    backend/chunk_debug/<source_stem>_<timestamp>/chunks.txt  so you can open
    the folder and inspect chunking quality without running a query.

    Returns the path of the file that was written.
    """
    import datetime

    # Resolve the debug folder relative to this file's location (backend/)
    backend_dir = Path(__file__).parent.parent
    debug_root = backend_dir / "chunk_debug"

    safe_stem = re.sub(r"[^\w\-.]", "_", Path(source_name).stem)[:60]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = debug_root / f"{safe_stem}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunks.txt"

    divider = "═" * 72

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"Document : {source_name}\n")
        fh.write(f"Generated: {datetime.datetime.now().isoformat()}\n")
        fh.write(f"Chunks   : {len(chunks)}\n")
        fh.write(f"{divider}\n\n")

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            content = chunk.get("content", "")
            idx = meta.get("chunk_index", "?")
            fh.write(f"┌─ CHUNK {idx} / {len(chunks)}")

            # Append tags on the header line for quick visual scan
            tags = []
            if meta.get("is_table"):
                tags.append("TABLE")
            if meta.get("is_image_summary"):
                tags.append("IMAGE_SUMMARY")
            if tags:
                fh.write(f"  [{', '.join(tags)}]")
            fh.write("\n")

            # Metadata block
            fh.write("│ Metadata\n")
            for key, val in sorted(meta.items()):
                fh.write(f"│   {key:<20} : {val}\n")

            # Content block
            fh.write("│ Content\n")
            for line in content.splitlines():
                fh.write(f"│   {line}\n")

            fh.write(f"└{'─' * 70}\n\n")

    logger.info("[CHUNK_DEBUG] Saved %d chunks → %s", len(chunks), out_path)
    return str(out_path)
