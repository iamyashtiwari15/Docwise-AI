from agents.rag_agent.role_llm_loader import get_llm


class AdvancedQueryMerger:
    @staticmethod
    def merge(current_query, chat_history, max_history=4):
        """
        Merge follow-up questions with recent history only when needed.
        Args:
            current_query (str): The latest user query.
            chat_history (list): List of previous exchanges (dicts with 'role' and 'content').
            max_history (int): Number of previous turns to include.
        Returns:
            str: Rewritten, self-contained query string.
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
            (msg["content"] for msg in reversed(relevant_history) if msg.get("role") == "user"),
            "",
        )
        if not last_user_message:
            return current_query

        if len(current_query.split()) <= 6:
            return f"{current_query}\n\nRelated context: {last_user_message}"

        history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in relevant_history])
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
