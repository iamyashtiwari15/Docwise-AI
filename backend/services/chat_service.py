import logging
import os
import time
import uuid
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agents.document_ingestion import is_document_upload
from agents.uploaded_document_store import uploaded_document_store

logger = logging.getLogger(__name__)


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
    def _graph_config(self, session_id: str) -> dict:
        """LangGraph run config — identifies the thread so InMemorySaver restores state."""
        return {"configurable": {"thread_id": session_id}}

    def _prior_messages(self, session_id: str) -> list[BaseMessage]:
        """Read the accumulated message history from the checkpointer before this turn."""
        from agents.agent_decision import assistant_graph
        state = assistant_graph.get_state(self._graph_config(session_id))
        if state and state.values:
            return list(state.values.get("messages", []))
        return []

    def _append_to_graph(self, session_id: str, message: BaseMessage) -> None:
        """Inject a message into the checkpointed graph state outside of a graph run."""
        from agents.agent_decision import assistant_graph
        assistant_graph.update_state(self._graph_config(session_id), {"messages": [message]})

    def _invoke_agent_graph(
        self,
        user_input: Optional[str],
        session_id: str,
        image_bytes: Optional[bytes] = None,
        image_type: Optional[str] = None,
    ) -> dict[str, Any]:
        from agents.agent_decision import assistant_graph

        # Snapshot history before this turn — used by retrieval & query rewriter
        chat_history = self._prior_messages(session_id)
        input_type = "image" if image_bytes else "text"
        image_path = None

        logger.debug(
            "[GRAPH] Invoking agent graph | session=%s | input_type=%s | history_len=%d",
            session_id, input_type, len(chat_history),
        )

        if image_bytes and image_type:
            extension = image_type.split("/")[-1]
            image_path = f"temp_{uuid.uuid4()}.{extension}"
            with open(image_path, "wb") as file_handle:
                file_handle.write(image_bytes)

        # Pass only the NEW HumanMessage in "messages" — the add_messages reducer
        # in GraphState appends it to the checkpointer's accumulated history.
        new_messages: list[BaseMessage] = []
        if user_input:
            new_messages.append(HumanMessage(content=user_input))

        input_state = {
            "input": user_input or "",
            "session_id": session_id,
            "image": image_bytes,
            "image_path": image_path,
            "image_type": image_type or "",
            "input_type": input_type,
            "agent_name": "",        # reset transient fields each turn
            "response": "",
            "workflow_response": {},
            "involved_agents": [],
            "bypass_guardrails": False,
            "messages": new_messages,
            "chat_history": chat_history,  # prior turns as List[BaseMessage]
        }

        t0 = time.perf_counter()
        try:
            output = assistant_graph.invoke(input_state, self._graph_config(session_id))
        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

        latency_ms = (time.perf_counter() - t0) * 1000
        agents_used = output.get("involved_agents", [])
        route = output.get("workflow_response", {}).get("query_type", "unknown")
        logger.info(
            "[GRAPH] Completed | session=%s | agents=%s | route=%s | latency=%.0fms",
            session_id, agents_used, route, latency_ms,
        )
        return output

    def process_text_message(self, session_id: str, message: str) -> dict[str, Any]:
        agent_output = self._invoke_agent_graph(message, session_id)
        return _extract_response_payload(agent_output)

    def process_image_message(self, session_id: str, message: str, file_bytes: bytes, content_type: str | None) -> dict[str, Any]:
        agent_output = self._invoke_agent_graph(message, session_id, image_bytes=file_bytes, image_type=content_type)
        return _extract_response_payload(agent_output)

    def ingest_document_background(
        self,
        session_id: str,
        filename: str,
        file_bytes: bytes,
        content_type: str | None,
    ) -> None:
        """
        Background worker: chunk + embed a document and record ingestion status.
        Called by the /ingest endpoint immediately when user attaches a file.
        """
        from agents.uploaded_document_store import ingestion_tracker
        logger.info("[INGEST] Background ingestion started | session=%s | filename=%s", session_id, filename)
        try:
            result = uploaded_document_store.add_file(session_id, filename, file_bytes, content_type)
            self._append_to_graph(session_id, HumanMessage(content=f"[Uploaded file: {filename}]"))
            ingestion_tracker.finish(session_id, filename, result["chunk_count"])
            logger.info(
                "[INGEST] Done | session=%s | filename=%s | chunks=%d",
                session_id, filename, result["chunk_count"],
            )
        except Exception as exc:
            from agents.uploaded_document_store import ingestion_tracker
            ingestion_tracker.fail(session_id, filename, str(exc))
            logger.error("[INGEST] Failed | session=%s | filename=%s | error=%s", session_id, filename, exc)

    def process_upload(
        self,
        session_id: str,
        filename: str,
        file_bytes: bytes,
        content_type: str | None,
        message: str = "",
    ) -> dict[str, Any]:
        if is_document_upload(filename, content_type):
            from agents.uploaded_document_store import IngestionStatus, ingestion_tracker

            status = ingestion_tracker.get_status(session_id, filename)

            if status == IngestionStatus.DONE:
                # Already indexed by background /ingest — skip re-ingestion
                logger.info("[UPLOAD] File already indexed via /ingest | session=%s | file=%s", session_id, filename)
                chunk_count = uploaded_document_store.chunk_count(session_id)

            elif status == IngestionStatus.PENDING:
                # Background indexing is in progress — wait for it
                logger.info("[UPLOAD] Waiting for background ingestion | session=%s | file=%s", session_id, filename)
                entry = ingestion_tracker.wait(session_id, filename, timeout=120.0)
                if entry and entry.status == IngestionStatus.ERROR:
                    return {
                        "response": f"⚠️ Failed to index **{filename}**: {entry.error}",
                        "sources": [], "confidence": 0.0, "query_type": "document",
                    }
                chunk_count = entry.chunk_count if entry else 0

            else:
                # /ingest was never called (e.g. direct API use) — ingest synchronously
                ingestion_tracker.start(session_id, filename)
                try:
                    result = uploaded_document_store.add_file(session_id, filename, file_bytes, content_type)
                    chunk_count = result["chunk_count"]
                    self._append_to_graph(session_id, HumanMessage(content=f"[Uploaded file: {filename}]"))
                    ingestion_tracker.finish(session_id, filename, chunk_count)
                except Exception as exc:
                    ingestion_tracker.fail(session_id, filename, str(exc))
                    return {
                        "response": f"⚠️ Failed to index **{filename}**: {exc}",
                        "sources": [], "confidence": 0.0, "query_type": "document",
                    }

            if message.strip():
                return self.process_text_message(session_id, message)

            response_text = (
                f"Uploaded **{filename}** and indexed {chunk_count} chunks. "
                "You can now ask questions about the uploaded document."
            )
            self._append_to_graph(session_id, AIMessage(content=response_text))
            return {"response": response_text, "sources": [], "confidence": 1.0, "query_type": "document"}

        return self.process_image_message(session_id, message, file_bytes, content_type)

    def get_history(self, session_id: str) -> dict[str, Any]:
        messages = self._prior_messages(session_id)
        # Serialise BaseMessage objects to dicts for the API response
        history = [
            {"role": "user" if msg.type == "human" else "assistant", "content": str(msg.content)}
            for msg in messages
            if hasattr(msg, "type") and hasattr(msg, "content")
        ]
        return {
            "history": history,
            "uploaded_files": uploaded_document_store.list_files(session_id),
        }


chat_service = ChatService()
