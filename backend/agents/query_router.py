import logging
import re
from typing import Literal, TypedDict

logger = logging.getLogger(__name__)


class RouteDecision(TypedDict):
    query_type: Literal["document", "web", "general"]
    reason: str


class QueryRouter:
    GENERAL_PATTERNS = [
        re.compile(r"^\s*(hi|hello|hey|thanks|thank you|bye|goodbye)\b", re.IGNORECASE),
        re.compile(r"\b(how are you|what can you do|who are you)\b", re.IGNORECASE),
        # Meta-conversational: user is asking about the current conversation itself
        re.compile(r"\b(the conversation|our conversation|our chat|our discussion|we discussed|we talked about)\b", re.IGNORECASE),
        re.compile(r"\b(summarize|summarise|summary|summaries|recap|recapitulate)\s+(the\s+)?(conversation|chat|discussion|above|so far|till now|history)\b", re.IGNORECASE),
    ]
    WEB_PATTERNS = [
        re.compile(r"\b(latest|current|today|news|weather|score|won|price|stock|market|live)\b", re.IGNORECASE),
        re.compile(r"\b(2024|2025|2026)\b"),
        re.compile(r"\b(ipl|world cup|bitcoin|share price|election)\b", re.IGNORECASE),
    ]
    DOCUMENT_PATTERNS = [
        re.compile(r"\b(document|file|pdf|docx|upload|uploaded|attachment|attached)\b", re.IGNORECASE),
        re.compile(r"\b(contract|policy|report|manual|resume|invoice|statement|proposal|agreement)\b", re.IGNORECASE),
        # Match all inflections: summarize, summarise, summarizing, summary, summaries, summarization
        re.compile(r"\bsummar", re.IGNORECASE),
        re.compile(r"\b(explain|review|analyze|analyse)\b", re.IGNORECASE),
    ]

    def route_query(self, query: str, has_uploaded_documents: bool) -> RouteDecision:
        normalized_query = (query or "").strip()
        if not normalized_query:
            decision: RouteDecision = {"query_type": "general", "reason": "empty query"}
            logger.debug("[ROUTER] %s | query=(empty)", decision["query_type"].upper())
            return decision

        for pattern in self.GENERAL_PATTERNS:
            if pattern.search(normalized_query):
                decision = {"query_type": "general", "reason": "greeting or casual chat"}
                logger.debug("[ROUTER] GENERAL | reason=%s | query=%.60r", decision["reason"], normalized_query)
                return decision

        for pattern in self.DOCUMENT_PATTERNS:
            if pattern.search(normalized_query):
                decision = {"query_type": "document", "reason": "query references uploaded documents"}
                logger.debug("[ROUTER] DOCUMENT | reason=%s | query=%.60r", decision["reason"], normalized_query)
                return decision

        for pattern in self.WEB_PATTERNS:
            if pattern.search(normalized_query):
                decision = {"query_type": "web", "reason": "query needs current or public web information"}
                logger.debug("[ROUTER] WEB | reason=%s | query=%.60r", decision["reason"], normalized_query)
                return decision

        if has_uploaded_documents:
            decision = {"query_type": "document", "reason": "documents are available, so document retrieval is preferred"}
            logger.debug("[ROUTER] DOCUMENT | reason=%s | query=%.60r", decision["reason"], normalized_query)
            return decision

        decision = {"query_type": "web", "reason": "no uploaded documents available, defaulting to web search"}
        logger.debug("[ROUTER] WEB | reason=%s | query=%.60r", decision["reason"], normalized_query)
        return decision
