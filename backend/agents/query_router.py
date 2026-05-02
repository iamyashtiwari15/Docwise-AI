import re
from typing import Literal, TypedDict


class RouteDecision(TypedDict):
    query_type: Literal["document", "web", "general"]
    reason: str


class QueryRouter:
    GENERAL_PATTERNS = [
        re.compile(r"^\s*(hi|hello|hey|thanks|thank you|bye|goodbye)\b", re.IGNORECASE),
        re.compile(r"\b(how are you|what can you do|who are you)\b", re.IGNORECASE),
    ]
    WEB_PATTERNS = [
        re.compile(r"\b(latest|current|today|now|news|weather|score|won|price|stock|market|live)\b", re.IGNORECASE),
        re.compile(r"\b(2024|2025|2026)\b"),
        re.compile(r"\b(ipl|world cup|bitcoin|share price|election)\b", re.IGNORECASE),
    ]
    DOCUMENT_PATTERNS = [
        re.compile(r"\b(document|file|pdf|docx|upload|uploaded|attachment|attached)\b", re.IGNORECASE),
        re.compile(r"\b(contract|policy|report|manual|resume|invoice|statement|proposal|agreement)\b", re.IGNORECASE),
        re.compile(r"\b(summarize|summarise|explain|review|analyze|analyse)\b", re.IGNORECASE),
    ]

    def route_query(self, query: str, has_uploaded_documents: bool) -> RouteDecision:
        normalized_query = (query or "").strip()
        if not normalized_query:
            return {"query_type": "general", "reason": "empty query"}

        for pattern in self.GENERAL_PATTERNS:
            if pattern.search(normalized_query):
                return {"query_type": "general", "reason": "greeting or casual chat"}

        for pattern in self.DOCUMENT_PATTERNS:
            if pattern.search(normalized_query):
                return {"query_type": "document", "reason": "query references uploaded documents"}

        for pattern in self.WEB_PATTERNS:
            if pattern.search(normalized_query):
                return {"query_type": "web", "reason": "query needs current or public web information"}

        if has_uploaded_documents:
            return {"query_type": "document", "reason": "documents are available, so document retrieval is preferred"}

        return {"query_type": "web", "reason": "no uploaded documents available, defaulting to web search"}
