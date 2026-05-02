from typing import Any

from pydantic import BaseModel, Field


class MessagePayload(BaseModel):
    role: str
    content: str


class HistoryRequest(BaseModel):
    session_id: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    response: str
    sources: list[Any] = Field(default_factory=list)
    confidence: float | None = None
    query_type: str | None = None


class HistoryResponse(BaseModel):
    history: list[MessagePayload] = Field(default_factory=list)
    uploaded_files: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
