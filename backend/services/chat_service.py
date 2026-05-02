import ast
import os
import uuid
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agents.document_ingestion import is_document_upload
from agents.uploaded_document_store import uploaded_document_store
from services.session_store import session_store


def convert_history(messages: list[BaseMessage]) -> list[dict[str, str]]:
    history = []
    for message in messages:
        if hasattr(message, "type") and hasattr(message, "content"):
            role = "user" if message.type == "human" else "assistant"
            history.append({"role": role, "content": str(message.content)})
    return history


def _normalize_message(message: BaseMessage) -> BaseMessage:
    if not hasattr(message, "type") or not hasattr(message, "content"):
        return message

    if message.type != "ai":
        return message

    content = message.content
    if isinstance(content, dict) and "response" in content:
        return AIMessage(content=content["response"])

    if isinstance(content, str) and content.strip().startswith("{"):
        try:
            import json

            payload = json.loads(content)
            if isinstance(payload, dict) and "response" in payload:
                return AIMessage(content=payload["response"])
        except Exception:
            try:
                payload = ast.literal_eval(content)
                if isinstance(payload, dict) and "response" in payload:
                    return AIMessage(content=payload["response"])
            except Exception:
                pass

    return message


def _extract_response_payload(agent_output: dict[str, Any]) -> dict[str, Any]:
    response_payload = agent_output.get("response", {})
    if isinstance(response_payload, dict):
        return {
            "response": response_payload.get("response", "Error: Could not find a response."),
            "sources": response_payload.get("sources", []),
            "confidence": response_payload.get("confidence", 0.0),
            "query_type": response_payload.get("query_type"),
        }

    return {
        "response": str(response_payload),
        "sources": [],
        "confidence": 0.0,
        "query_type": None,
    }


class ChatService:
    def _invoke_agent_graph(
        self,
        user_input: Optional[str],
        session_id: str,
        image_bytes: Optional[bytes] = None,
        image_type: Optional[str] = None,
    ) -> dict[str, Any]:
        messages = session_store.get_messages(session_id)
        chat_history = convert_history(messages)
        input_type = "image" if image_bytes else "text"
        image_path = None

        if image_bytes and image_type:
            extension = image_type.split("/")[-1]
            image_path = f"temp_{uuid.uuid4()}.{extension}"
            with open(image_path, "wb") as file_handle:
                file_handle.write(image_bytes)

        if user_input:
            messages.append(HumanMessage(content=user_input))

        input_state = {
            "input": user_input or "",
            "session_id": session_id,
            "image": image_bytes,
            "image_path": image_path,
            "image_type": image_type or "",
            "input_type": input_type,
            "agent_name": "",
            "response": "",
            "workflow_response": {},
            "involved_agents": [],
            "bypass_guardrails": False,
            "messages": messages,
            "chat_history": chat_history,
        }

        try:
            from agents.agent_decision import assistant_graph

            output = assistant_graph.invoke(input_state)
        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

        normalized_messages = [_normalize_message(message) for message in output.get("messages", messages)]
        session_store.set_messages(session_id, normalized_messages)
        return output

    def process_text_message(self, session_id: str, message: str) -> dict[str, Any]:
        agent_output = self._invoke_agent_graph(message, session_id)
        return _extract_response_payload(agent_output)

    def process_image_message(self, session_id: str, message: str, file_bytes: bytes, content_type: str | None) -> dict[str, Any]:
        agent_output = self._invoke_agent_graph(message, session_id, image_bytes=file_bytes, image_type=content_type)
        return _extract_response_payload(agent_output)

    def process_upload(
        self,
        session_id: str,
        filename: str,
        file_bytes: bytes,
        content_type: str | None,
        message: str = "",
    ) -> dict[str, Any]:
        if is_document_upload(filename, content_type):
            ingestion_result = uploaded_document_store.add_file(session_id, filename, file_bytes, content_type)
            session_store.append_message(session_id, HumanMessage(content=f"[Uploaded file: {filename}]"))

            if message.strip():
                return self.process_text_message(session_id, message)

            response_text = (
                f"Uploaded **{filename}** and indexed {ingestion_result['chunk_count']} chunks. "
                "You can now ask questions about the uploaded document."
            )
            session_store.append_message(session_id, AIMessage(content=response_text))
            return {"response": response_text, "sources": [], "confidence": 1.0, "query_type": "document"}

        return self.process_image_message(session_id, message, file_bytes, content_type)

    def get_history(self, session_id: str) -> dict[str, Any]:
        messages = session_store.get_messages(session_id)
        return {
            "history": convert_history(messages),
            "uploaded_files": uploaded_document_store.list_files(session_id),
        }


chat_service = ChatService()
