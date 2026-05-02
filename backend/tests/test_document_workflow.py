import unittest
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from agents.document_ingestion import chunk_document_text, is_document_upload
from agents.query_router import QueryRouter
from agents.retrieval_utils import generate_query_variants
from agents.uploaded_document_store import UploadedDocumentStore


class FakeEmbeddingModel:
    FEATURES = ("leave", "policy", "refund", "timeline", "complaint", "ipl")

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            lower_text = text.lower()
            embeddings.append([1.0 if feature in lower_text else 0.0 for feature in self.FEATURES])
        return embeddings

    def embed_query(self, text):
        lower_text = text.lower()
        return [1.0 if feature in lower_text else 0.0 for feature in self.FEATURES]


class FakeLLMResult:
    def __init__(self, content):
        self.content = content


class FakeMultiQueryLLM:
    def __init__(self, content):
        self.content = content
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return FakeLLMResult(self.content)


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

    def test_chunk_document_text_splits_long_paragraph_without_fragment_overlap(self):
        text = (
            "The refund policy explains eligibility requirements for customers. "
            "The complaint timeline starts after the refund request is accepted. "
            "The authority must respond within the stated service window."
        )
        chunks = chunk_document_text(text, "policy.txt", chunk_size=80, overlap=20)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk["content"].strip() for chunk in chunks))
        self.assertFalse(chunks[1]["content"].startswith("pond within"))

    def test_query_router_prefers_document_when_docs_exist(self):
        router = QueryRouter()
        decision = router.route_query("What is the leave policy?", has_uploaded_documents=True)
        self.assertEqual(decision["query_type"], "document")

    def test_query_router_routes_live_question_to_web(self):
        router = QueryRouter()
        decision = router.route_query("Who won the IPL in 2025?", has_uploaded_documents=True)
        self.assertEqual(decision["query_type"], "web")

    def test_query_router_does_not_send_conversation_summary_to_web(self):
        router = QueryRouter()
        # "now" is a filler here, not a web signal; "summaries" is an inflection of "summarize"
        decision = router.route_query("now summaries the conversation", has_uploaded_documents=True)
        self.assertNotEqual(decision["query_type"], "web")

    def test_query_router_handles_summar_variants(self):
        router = QueryRouter()
        for query in ["summarize this document", "give me a summary", "summaries the key points"]:
            with self.subTest(query=query):
                decision = router.route_query(query, has_uploaded_documents=True)
                self.assertEqual(decision["query_type"], "document")

    def test_query_router_now_alone_does_not_trigger_web(self):
        router = QueryRouter()
        decision = router.route_query("now explain the refund policy", has_uploaded_documents=True)
        self.assertEqual(decision["query_type"], "document")

    def test_uploaded_document_store_retrieves_relevant_chunk(self):
        store = UploadedDocumentStore(embedding_model=FakeEmbeddingModel())
        store.add_file("session-1", "policy.txt", b"Leave policy allows 20 annual leaves.", "text/plain")
        results = store.retrieve("session-1", "What is the leave policy?")
        self.assertTrue(results)
        self.assertIn("Leave policy", results[0]["content"])
        self.assertGreater(results[0]["score"], 0)

    def test_generate_query_variants_for_compound_question(self):
        variants = generate_query_variants(
            "Explain the leave policy and the refund timeline for complaints.",
            max_variants=4,
        )
        self.assertGreaterEqual(len(variants), 2)
        lowered_variants = [variant.lower() for variant in variants]
        self.assertTrue(any("leave policy" in variant for variant in lowered_variants))
        self.assertTrue(any("refund timeline" in variant for variant in lowered_variants))

    def test_generate_query_variants_uses_llm_when_available(self):
        llm = FakeMultiQueryLLM(
            "leave policy annual leave entitlement\nrefund timeline complaint resolution window"
        )
        variants = generate_query_variants(
            "Explain the leave policy and the refund timeline for complaints.",
            max_variants=4,
            llm=llm,
        )

        self.assertTrue(llm.prompts)
        lowered_variants = [variant.lower() for variant in variants]
        self.assertTrue(any("annual leave entitlement" in variant for variant in lowered_variants))
        self.assertTrue(any("complaint resolution window" in variant for variant in lowered_variants))

    def test_uploaded_document_store_uses_multi_query_rerank(self):
        store = UploadedDocumentStore(embedding_model=FakeEmbeddingModel())
        store.add_file("session-1", "policy.txt", b"Leave policy allows 20 annual leaves.", "text/plain")
        store.add_file("session-1", "refund.txt", b"Complaint refund timeline is 7 days from approval.", "text/plain")

        results = store.retrieve("session-1", "Explain the leave policy and refund timeline", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual({result["source"] for result in results}, {"policy.txt", "refund.txt"})
        self.assertTrue(all("combined_score" in result for result in results))
        self.assertTrue(all("rerank_score" in result for result in results))

    def test_uploaded_document_store_uses_llm_multi_query_variants(self):
        llm = FakeMultiQueryLLM(
            "leave policy annual leave entitlement\nrefund timeline complaint resolution window"
        )
        store = UploadedDocumentStore(embedding_model=FakeEmbeddingModel(), query_variant_llm=llm)
        store.add_file("session-1", "policy.txt", b"Leave policy allows 20 annual leaves.", "text/plain")
        store.add_file("session-1", "refund.txt", b"Complaint refund timeline is 7 days from approval.", "text/plain")

        results = store.retrieve("session-1", "Explain the leave policy and the refund timeline for complaints", top_k=2)

        self.assertTrue(llm.prompts)
        self.assertEqual(len(results), 2)
        self.assertEqual({result["source"] for result in results}, {"policy.txt", "refund.txt"})


if __name__ == "__main__":
    unittest.main()
