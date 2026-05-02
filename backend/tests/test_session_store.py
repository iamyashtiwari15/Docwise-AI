import sys
import unittest
from pathlib import Path

from langchain_core.messages import HumanMessage

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.session_store import SessionMessageStore


class SessionStoreTests(unittest.TestCase):
    def test_store_trims_history(self):
        store = SessionMessageStore(max_messages=2)
        store.set_messages(
            "session-1",
            [HumanMessage(content="one"), HumanMessage(content="two"), HumanMessage(content="three")],
        )

        messages = store.get_messages("session-1")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].content, "two")
        self.assertEqual(messages[1].content, "three")

    def test_append_message_creates_session(self):
        store = SessionMessageStore(max_messages=3)
        store.append_message("session-2", HumanMessage(content="hello"))
        messages = store.get_messages("session-2")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].content, "hello")


if __name__ == "__main__":
    unittest.main()
