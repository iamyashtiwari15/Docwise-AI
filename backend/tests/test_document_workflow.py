import unittest
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.document_ingestion import chunk_document_text, is_document_upload
from agents.query_router import QueryRouter
from agents.uploaded_document_store import UploadedDocumentStore


class FakeEmbeddingModel:
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            lower_text = text.lower()
            embeddings.append(
                [
                    1.0 if "leave" in lower_text else 0.0,
                    1.0 if "policy" in lower_text else 0.0,
                    1.0 if "ipl" in lower_text else 0.0,
                ]
            )
        return embeddings

    def embed_query(self, text):
        lower_text = text.lower()
        return [
            1.0 if "leave" in lower_text else 0.0,
            1.0 if "policy" in lower_text else 0.0,
            1.0 if "ipl" in lower_text else 0.0,
        ]


class DocumentWorkflowTests(unittest.TestCase):
    def test_document_upload_detection_accepts_pdf_and_docx(self):
        self.assertTrue(is_document_upload("policy.pdf", "application/pdf"))
        self.assertTrue(
            is_document_upload(
                "handbook.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        )

    def test_chunk_document_text_produces_chunks(self):
        chunks = chunk_document_text("Para one\nPara two\nPara three", "sample.txt", chunk_size=10, overlap=0)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["metadata"]["source"], "sample.txt")

    def test_query_router_prefers_document_when_docs_exist(self):
        router = QueryRouter()
        decision = router.route_query("What is the leave policy?", has_uploaded_documents=True)
        self.assertEqual(decision["query_type"], "document")

    def test_query_router_routes_live_question_to_web(self):
        router = QueryRouter()
        decision = router.route_query("Who won the IPL in 2025?", has_uploaded_documents=True)
        self.assertEqual(decision["query_type"], "web")

    def test_uploaded_document_store_retrieves_relevant_chunk(self):
        store = UploadedDocumentStore(embedding_model=FakeEmbeddingModel())
        store.add_file("session-1", "policy.txt", b"Leave policy allows 20 annual leaves.", "text/plain")
        results = store.retrieve("session-1", "What is the leave policy?")
        self.assertTrue(results)
        self.assertIn("Leave policy", results[0]["content"])
        self.assertGreater(results[0]["score"], 0)


if __name__ == "__main__":
    unittest.main()
