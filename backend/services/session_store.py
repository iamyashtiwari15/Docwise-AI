from threading import Lock

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage

from core.config import get_settings


class SessionMessageStore:
    """
    Per-session chat history backed by LangChain's InMemoryChatMessageHistory.
    Provides get_history() for direct LangChain chain integration and keeps
    the same get_messages / set_messages / append_message interface for the
    rest of the codebase.
    """

    def __init__(self, max_messages: int | None = None):
        settings = get_settings()
        self.max_messages = max_messages or settings.session_max_messages
        self._lock = Lock()
        self._histories: dict[str, InMemoryChatMessageHistory] = {}

    # ── LangChain-native accessor ──────────────────────────────────────────

    def get_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Return the InMemoryChatMessageHistory for a session (creates on first access)."""
        with self._lock:
            if session_id not in self._histories:
                self._histories[session_id] = InMemoryChatMessageHistory()
            return self._histories[session_id]

    # ── Compatibility API (used throughout the codebase) ──────────────────

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        """Return a trimmed copy of all messages for the session."""
        return list(self.get_history(session_id).messages)[-self.max_messages :]

    def set_messages(self, session_id: str, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Replace all messages for the session (trims to max_messages)."""
        trimmed = list(messages)[-self.max_messages :]
        history = self.get_history(session_id)
        history.clear()
        for msg in trimmed:
            history.add_message(msg)
        return trimmed

    def append_message(self, session_id: str, message: BaseMessage) -> list[BaseMessage]:
        """Append a single message and trim if necessary."""
        history = self.get_history(session_id)
        history.add_message(message)
        if len(history.messages) > self.max_messages:
            trimmed = list(history.messages)[-self.max_messages :]
            history.clear()
            for msg in trimmed:
                history.add_message(msg)
        return list(history.messages)


session_store = SessionMessageStore()
