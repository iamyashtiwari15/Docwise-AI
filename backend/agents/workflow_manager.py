import logging
import time
from typing import TypedDict, Optional, Dict, Any, List

from langchain_core.messages import BaseMessage


class WorkflowResponse(TypedDict, total=False):
    response: str
    sources: List[str]
    confidence: float
    verification_result: Any

from agents.rag_agent.response_generator import ResponseGenerator
from agents.web_search.tavily_search import FlexibleTavilySearchAgent
from agents.query_merger import AdvancedQueryMerger
from agents.query_router import QueryRouter
from agents.uploaded_document_store import uploaded_document_store
from core.config import get_settings

logger = logging.getLogger(__name__)

class WorkflowManager:
    """
    Orchestrates document, web, and general-chat routing for the assistant.
    """
    def __init__(self, db_path):
        self.db_path = db_path
        self.router = QueryRouter()
        self.response_generator = ResponseGenerator()
        self.web_search_agent = FlexibleTavilySearchAgent()
        self.settings = get_settings()

    def _build_web_docs(self, query: str) -> List[Dict[str, Any]]:
        t0 = time.perf_counter()
        results = self.web_search_agent.search_results(query, trusted_sites_only=False)
        latency_ms = (time.perf_counter() - t0) * 1000
        if not results:
            logger.warning("[WEB] Search returned no usable results | latency=%.0fms", latency_ms)
            return []
        logger.info("[WEB] Search completed | results=%d | latency=%.0fms", len(results), latency_ms)
        return [
            {
                "content": f"Title: {r['title']}\nURL: {r['url']}\n\n{r['content']}",
                "score": 0.7,
                "source": r["title"],
                "source_path": r["url"],
                "metadata": {
                    "source": r["title"],
                    "source_path": r["url"],
                    "score": 0.7,
                },
            }
            for r in results
        ]

    def _top_score(self, docs: List[Dict[str, Any]]) -> float:
        if not docs:
            return 0.0
        score_keys = ("combined_score", "rerank_score", "score")
        top_score = 0.0
        for doc in docs:
            for key in score_keys:
                if key in doc:
                    top_score = max(top_score, float(doc[key]))
                    break
                if key in doc.get("metadata", {}):
                    top_score = max(top_score, float(doc["metadata"][key]))
                    break
        return top_score

    def _generate_response(self, query: str, docs: List[Dict[str, Any]], chat_history: Optional[List[BaseMessage]]) -> WorkflowResponse:
        t0 = time.perf_counter()
        logger.debug("[LLM] Starting response generation | docs=%d", len(docs))
        try:
            response = self.response_generator.generate_response(
                query=query,
                retrieved_docs=docs,
                chat_history=chat_history
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            if not response or not response.get("response"):
                logger.warning("[LLM] Empty response generated | latency=%.0fms", latency_ms)
                return {"response": "Sorry, no information was found for your query. Please try rephrasing or ask about another topic.", "sources": [], "confidence": 0.0}
            logger.info("[LLM] Response generated | confidence=%.2f | latency=%.0fms", response.get("confidence", 0.0), latency_ms)
            return response
        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error("[LLM] Response generation failed | latency=%.0fms | error=%s", latency_ms, e)
            return {"response": "Sorry, something went wrong. Please try again later.", "sources": [], "confidence": 0.0}

    def process_query(
        self,
        query: str,
        image_path: Optional[str] = None,
        chat_history: Optional[List[BaseMessage]] = None,
        image_context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> WorkflowResponse:
        """
        Process a query by routing it to uploaded-document retrieval, web search, or general chat.
        """
        t_total = time.perf_counter()
        logger.info("[QUERY] Received | session=%s | query=%.80r", session_id, query)

        merged_query = AdvancedQueryMerger.merge(query, chat_history or [])
        if image_context and "❌" not in (image_context or ""):
            merged_query = f"{merged_query}\n\nImage context:\n{image_context}"

        has_uploaded_documents = bool(session_id and uploaded_document_store.has_documents(session_id))

        t_route = time.perf_counter()
        route = self.router.route_query(merged_query, has_uploaded_documents)
        logger.info(
            "[ROUTE] Decision → %s | reason=%s | has_docs=%s | latency=%.0fms",
            route["query_type"].upper(),
            route["reason"],
            has_uploaded_documents,
            (time.perf_counter() - t_route) * 1000,
        )

        # ── GENERAL (greeting / chitchat) ──────────────────────────────────
        if route["query_type"] == "general":
            logger.info("[ROUTE] → GENERAL CHAT | no retrieval needed")
            return {
                "response": "Happy to help. Ask a question, upload a PDF or DOCX for document Q&A, or ask for current information from the web.",
                "sources": [],
                "confidence": 1.0,
                "query_type": "general",
            }

        # ── DOCUMENT ───────────────────────────────────────────────────────
        if route["query_type"] == "document":
            if not has_uploaded_documents or not session_id:
                logger.info("[ROUTE] → DOCUMENT requested but no documents uploaded | session=%s", session_id)
                return {
                    "response": "Upload a PDF or DOCX first, then ask a question about the document. If you want general current information, ask directly and I will use web search.",
                    "sources": [],
                    "confidence": 0.0,
                    "query_type": "document",
                }

            t_retrieval = time.perf_counter()
            docs = uploaded_document_store.retrieve(session_id, merged_query, top_k=self.settings.rag_top_k)
            retrieval_ms = (time.perf_counter() - t_retrieval) * 1000
            top_score = self._top_score(docs)
            logger.info(
                "[ROUTE] → DOCUMENT retrieval | docs=%d | top_score=%.3f | threshold=%.2f | latency=%.0fms",
                len(docs), top_score, self.settings.document_relevance_threshold, retrieval_ms,
            )

            if docs and top_score >= self.settings.document_relevance_threshold:
                logger.info("[ROUTE] → RAG ✅ | score %.3f ≥ threshold %.2f", top_score, self.settings.document_relevance_threshold)
                response = self._generate_response(merged_query, docs, chat_history)
                response["query_type"] = "document"
                logger.info("[ROUTE] Final → RAG | total_latency=%.0fms", (time.perf_counter() - t_total) * 1000)
                return response

            # Confidence too low — fall back to web
            logger.info(
                "[ROUTE] → RAG score too low (%.3f < %.2f) — falling back to WEB",
                top_score, self.settings.document_relevance_threshold,
            )
            web_docs = self._build_web_docs(merged_query)
            if web_docs:
                response = self._generate_response(merged_query, web_docs, chat_history)
                response["query_type"] = "web"
                response["response"] = (
                    "I could not find a strong enough match in the uploaded documents, so I used web search instead.\n\n"
                    f"{response['response']}"
                )
                logger.info("[ROUTE] Final → WEB (fallback from RAG) | total_latency=%.0fms", (time.perf_counter() - t_total) * 1000)
                return response

            logger.warning("[ROUTE] Both RAG and WEB returned nothing | session=%s", session_id)
            return {
                "response": "I could not find a reliable answer in the uploaded documents, and web search did not return enough information.",
                "sources": [],
                "confidence": 0.0,
                "query_type": "document",
            }

        # ── WEB ────────────────────────────────────────────────────────────
        logger.info("[ROUTE] → WEB SEARCH")
        web_docs = self._build_web_docs(merged_query)
        if web_docs:
            response = self._generate_response(merged_query, web_docs, chat_history)
            response["query_type"] = "web"
            logger.info("[ROUTE] Final → WEB | total_latency=%.0fms", (time.perf_counter() - t_total) * 1000)
            return response

        logger.warning("[ROUTE] WEB search returned nothing | session=%s", session_id)
        return {
            "response": "I could not find enough reliable information to answer that right now.",
            "sources": [],
            "confidence": 0.0,
            "query_type": "web",
        }
