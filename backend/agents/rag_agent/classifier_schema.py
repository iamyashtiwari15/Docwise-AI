from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class QueryClassification(BaseModel):
    query_type: Literal[
        "greeting", "chitchat", "factual-lookup", "summarization",
        "comparison", "procedure", "analysis", "clarification", "general-info"
    ] = Field(..., description="Generic query type classification")
    topics: List[str] = Field(default_factory=list, description="List of relevant topics identified in the query.")
    has_actionable_request: bool = Field(..., description="True if the query contains an actionable request.")
    requires_external_sources: bool = Field(..., description="True if the query requires live or external information such as web search.")
    clarification_needed: bool = Field(default=False, description="True if the query is ambiguous or needs clarification.")
    clarification_question: Optional[str] = Field(default="", description="A clarifying question if the query is ambiguous; empty string otherwise.")
