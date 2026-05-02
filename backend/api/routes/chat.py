import asyncio

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse

from api.schemas import ChatResponse, HealthResponse, HistoryRequest, HistoryResponse
from core.config import get_settings
from services.chat_service import chat_service


settings = get_settings()
router = APIRouter(prefix=settings.api_prefix, tags=["assistant"])


@router.get("/health", response_model=HealthResponse)
async def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: str = Form(...), session_id: str = Form(...)) -> ChatResponse:
    try:
        payload = await asyncio.to_thread(chat_service.process_text_message, session_id, message)
        return ChatResponse(**payload)
    except Exception as error:
        return JSONResponse(content={"response": f"⚠️ Error: {str(error)}"}, status_code=500)


@router.post("/upload", response_model=ChatResponse)
async def upload_endpoint(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    message: str = Form(""),
):
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
        return ChatResponse(**payload)
    except Exception as error:
        return JSONResponse(content={"response": f"⚠️ Error: {str(error)}"}, status_code=500)


@router.post("/history", response_model=HistoryResponse)
async def history_endpoint(request: HistoryRequest) -> HistoryResponse:
    try:
        payload = await asyncio.to_thread(chat_service.get_history, request.session_id)
        return HistoryResponse(**payload)
    except Exception as error:
        return JSONResponse(content={"error": f"Error: {str(error)}"}, status_code=500)
