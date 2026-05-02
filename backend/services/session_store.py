from threading import Lock

from langchain_core.messages import BaseMessage

from core.config import get_settings


class SessionMessageStore:
    def __init__(self, max_messages: int | None = None):
        settings = get_settings()
        self.max_messages = max_messages or settings.session_max_messages
        self._lock = Lock()
        self._store: dict[str, list[BaseMessage]] = {}

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        with self._lock:
            return list(self._store.get(session_id, []))

    def set_messages(self, session_id: str, messages: list[BaseMessage]) -> list[BaseMessage]:
        trimmed_messages = list(messages)[-self.max_messages :]
        with self._lock:
            self._store[session_id] = trimmed_messages
        return list(trimmed_messages)

    def append_message(self, session_id: str, message: BaseMessage) -> list[BaseMessage]:
        messages = self.get_messages(session_id)
        messages.append(message)
        return self.set_messages(session_id, messages)


session_store = SessionMessageStore()
