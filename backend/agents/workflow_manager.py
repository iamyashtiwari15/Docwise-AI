import logging
from typing import TypedDict, Optional, Dict, Any, List


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
        web_results = self.web_search_agent.search(query, trusted_sites_only=False)
        if not web_results or web_results.startswith("No relevant results") or web_results.startswith("Error retrieving"):
            return []
        return [
            {
                "content": web_results,
                "score": 0.7,
                "source": "web_search",
                "source_path": "web_search",
                "metadata": {
                    "source": "web_search",
                    "source_path": "web_search",
                    "score": 0.7,
                },
            }
        ]

    def _top_score(self, docs: List[Dict[str, Any]]) -> float:
        if not docs:
            return 0.0
        return max(doc.get("score", doc.get("metadata", {}).get("score", 0.0)) for doc in docs)

    def _generate_response(self, query: str, docs: List[Dict[str, Any]], chat_history: Optional[List[Dict[str, str]]]) -> WorkflowResponse:
        """Generate the final response using the response generator."""
        logger.info(f"[LOG] Starting response generation for query: {query}")
        try:
            response = self.response_generator.generate_response(
                query=query,
                retrieved_docs=docs,
                chat_history=chat_history
            )
            logger.info(f"[LOG] Finished response generation. Response: {response}")
            if not response or not response.get("response"):
                logger.warning(f"[LOG] No response generated for query: {query}")
                return {"response": "Sorry, no information was found for your query. Please try rephrasing or ask about another topic.", "sources": [], "confidence": 0.0}
            return response
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {"response": "Sorry, something went wrong. Please try again later.", "sources": [], "confidence": 0.0}

    def process_query(
        self,
        query: str,
        image_path: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        image_context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> WorkflowResponse:
        """
        Process a query by routing it to uploaded-document retrieval, web search, or general chat.
        """
        logger.info(f"Received query: '{query}'")

        merged_query = AdvancedQueryMerger.merge(query, chat_history or [])
        if image_context and "❌" not in (image_context or ""):
            merged_query = f"{merged_query}\n\nImage context:\n{image_context}"

        has_uploaded_documents = bool(session_id and uploaded_document_store.has_documents(session_id))
        route = self.router.route_query(merged_query, has_uploaded_documents)
        logger.info(f"Route decision: {route}")

        if route["query_type"] == "general":
            return {
                "response": "Happy to help. Ask a question, upload a PDF or DOCX for document Q&A, or ask for current information from the web.",
                "sources": [],
                "confidence": 1.0,
                "query_type": "general",
            }

        if route["query_type"] == "document":
            if not has_uploaded_documents or not session_id:
                return {
                    "response": "Upload a PDF or DOCX first, then ask a question about the document. If you want general current information, ask directly and I will use web search.",
                    "sources": [],
                    "confidence": 0.0,
                    "query_type": "document",
                }

            docs = uploaded_document_store.retrieve(session_id, merged_query)
            if docs and self._top_score(docs) >= self.settings.document_relevance_threshold:
                response = self._generate_response(merged_query, docs, chat_history)
                response["query_type"] = "document"
                return response

            web_docs = self._build_web_docs(merged_query)
            if web_docs:
                response = self._generate_response(merged_query, web_docs, chat_history)
                response["query_type"] = "web"
                response["response"] = (
                    "I could not find a strong enough match in the uploaded documents, so I used web search instead.\n\n"
                    f"{response['response']}"
                )
                return response

            return {
                "response": "I could not find a reliable answer in the uploaded documents, and web search did not return enough information.",
                "sources": [],
                "confidence": 0.0,
                "query_type": "document",
            }

        web_docs = self._build_web_docs(merged_query)
        if web_docs:
            response = self._generate_response(merged_query, web_docs, chat_history)
            response["query_type"] = "web"
            return response

        return {
            "response": "I could not find enough reliable information to answer that right now.",
            "sources": [],
            "confidence": 0.0,
            "query_type": "web",
        }
