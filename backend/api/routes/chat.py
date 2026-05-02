import asyncio
import logging
import time

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from api.schemas import ChatResponse, HealthResponse, HistoryRequest, HistoryResponse
from core.config import get_settings
from services.chat_service import chat_service

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix=settings.api_prefix, tags=["assistant"])


@router.get("/health", response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: str = Form(...), session_id: str = Form(...)) -> ChatResponse:
    t0 = time.perf_counter()
    logger.info("[REQUEST] POST /chat | session=%s | message=%.80r", session_id, message)
    try:
        payload = await asyncio.to_thread(chat_service.process_text_message, session_id, message)
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[RESPONSE] POST /chat | session=%s | route=%s | latency=%.0fms",
            session_id,
            payload.get("query_type", "unknown"),
            latency_ms,
        )
        return ChatResponse(**payload)
    except Exception as error:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.error("[ERROR] POST /chat | session=%s | latency=%.0fms | error=%s", session_id, latency_ms, error)
        return JSONResponse(content={"response": f"⚠️ Error: {str(error)}"}, status_code=500)


@router.post("/upload", response_model=ChatResponse)
async def upload_endpoint(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    message: str = Form(""),
):
    t0 = time.perf_counter()
    logger.info("[REQUEST] POST /upload | session=%s | filename=%s | message=%.80r", session_id, file.filename, message)
    try:
        file_bytes = await file.read()
        filename = file.filename or "uploaded_file"
        payload = await asyncio.to_thread(
            chat_service.process_upload,
            session_id,
            filename,
            file_bytes,
            file.content_type,
            message,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "[RESPONSE] POST /upload | session=%s | route=%s | latency=%.0fms",
            session_id,
            payload.get("query_type", "unknown"),
            latency_ms,
        )
        return ChatResponse(**payload)
    except Exception as error:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.error("[ERROR] POST /upload | session=%s | latency=%.0fms | error=%s", session_id, latency_ms, error)
        return JSONResponse(content={"response": f"⚠️ Error: {str(error)}"}, status_code=500)


@router.post("/history", response_model=HistoryResponse)
async def history_endpoint(request: HistoryRequest) -> HistoryResponse:
    try:
        payload = await asyncio.to_thread(chat_service.get_history, request.session_id)
        return HistoryResponse(**payload)
    except Exception as error:
        logger.error("[ERROR] POST /history | session=%s | error=%s", request.session_id, error)
        return JSONResponse(content={"error": f"Error: {str(error)}"}, status_code=500)

