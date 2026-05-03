from typing import List
from langchain_core.messages import BaseMessage, HumanMessage
from agents.rag_agent.role_llm_loader import get_llm


class AdvancedQueryMerger:
    @staticmethod
    def merge(current_query: str, chat_history: List[BaseMessage], max_history: int = 4) -> str:
        """
        Rewrite a follow-up question into a self-contained query using prior turns.
        chat_history is a list of BaseMessage objects (HumanMessage / AIMessage).
        """
        current_query = (current_query or "").strip()
        if not current_query:
            return ""

        relevant_history = chat_history[-max_history:] if chat_history else []
        if not relevant_history:
            return current_query

        lowered_query = current_query.lower()
        follow_up_markers = ("this", "that", "it", "they", "them", "these", "those", "above", "earlier", "previous")
        if not any(marker in lowered_query.split() or marker in lowered_query for marker in follow_up_markers):
            return current_query

        last_user_message = next(
            (msg.content for msg in reversed(relevant_history) if isinstance(msg, HumanMessage)),
            "",
        )
        if not last_user_message:
            return current_query

        if len(current_query.split()) <= 6:
            return f"{current_query}\n\nRelated context: {last_user_message}"

        history_str = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in relevant_history
        ])
        prompt = f"""
Given the following chat history and the user's latest query, rewrite the query as a self-contained, contextually complete question. If the query is already self-contained, return it as is.

Chat History:
{history_str}

User's Latest Query:
{current_query}

Rewritten Query:
"""
        llm = get_llm(role="query_rewriter")
        result = llm.invoke(prompt)
        rewritten_query = result.content.strip() if hasattr(result, "content") else str(result).strip()
        return rewritten_query or current_query
