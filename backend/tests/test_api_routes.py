import sys
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from main import app


class ApiRouteTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_history_endpoint_for_new_session(self):
        response = self.client.post("/api/history", json={"session_id": "new-session"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"history": [], "uploaded_files": []})


if __name__ == "__main__":
    unittest.main()
