import logging
from typing import List, Dict, Any, Optional
from agents.rag_agent.role_llm_loader import get_llm


class ResponseGenerator:
    """
    Generates grounded responses based on retrieved document or web context.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.llm = get_llm()
        self.include_sources = True

    def _build_prompt(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        table_instructions = """
        Some of the retrieved information is presented in table format (e.g., timelines, refund rates, warranty durations). When using information from tables:
        1. Present tabular data using proper markdown table formatting with headers.
        2. Reformat the table for clarity, ensuring legal terms are clear.
        3. If adding an interpretation column (e.g., "Meaning"), mention it explicitly.
        4. Summarize legal requirements or obligations shown in the tables.
        """

        response_format_instructions = """Instructions:
        1. Answer the query based ONLY on the information provided in the context.
        2. If the context doesn't contain relevant information to answer the query, reply exactly in this format (and nothing else):
           {
               "Insufficient Information", "I don't have enough information."
           }
        3. Do not make up facts that are not supported by the provided context.
        4. Be concise, accurate, and avoid making assumptions not supported by the context.
        5. Format the answer with headings, subheadings, and tables (if applicable) in markdown.
        6. If numerical, date, or policy values are provided, use them exactly as they appear in the context.
        """

        history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history]) if chat_history else ""

        prompt = f"""You are a grounded assistant. Answer using only the supplied context from uploaded documents or trusted web results.

Here are the last few messages from our conversation:
{history_text}

The user has asked the following question:
{query}

I've retrieved the following information to help answer this question:
{context}

{table_instructions}
{response_format_instructions}

Based on the provided information, answer the user's question thoroughly but concisely.
If the information doesn't contain the answer, follow the insufficient information format exactly.
Do not provide any source link that is not present in the context.
"""

        return prompt

    def generate_response(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        query_classification: Optional[Any] = None,
        picture_paths: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        try:
            self.logger.info(f"[LOG] Starting LLM response generation for query: {query}")
            doc_texts_with_scores = []
            for doc in retrieved_docs:
                if isinstance(doc, tuple):
                    doc_texts_with_scores.append((doc[0].page_content, doc[1]))
                else:
                    doc_texts_with_scores.append((doc["content"], doc.get("score", doc.get("metadata", {}).get("score", 1.0))))

            doc_texts_with_scores.sort(key=lambda x: x[1], reverse=True)

            context_parts = []
            for text, score in doc_texts_with_scores:
                relevance_marker = "🔥 High Relevance" if score > 0.8 else "✓ Relevant" if score > 0.6 else "ℹ️ Context"
                context_parts.append(f"\n\n=== {relevance_marker} ===\n{text}")
            context = "\n".join(context_parts)

            prompt = self._build_prompt(query, context, chat_history)
            self.logger.info(f"[LOG] Prompt sent to LLM: {prompt}")

            result = self.llm.invoke(prompt)
            response_text = result.content.strip() if result and hasattr(result, "content") else None
            self.logger.info(f"[LOG] Raw LLM response: {response_text}")

            if not response_text:
                self.logger.warning(f"[LOG] LLM returned empty response for query: {query}")
                return {
                    "response": "Sorry, no information was found for your query. Please try rephrasing or ask about another topic.",
                    "sources": [],
                    "confidence": 0.0,
                }
            if response_text.startswith("{") and "Insufficient Information" in response_text:
                return {"response": response_text, "sources": [], "confidence": 0.0}

            sources = self._extract_sources(retrieved_docs) if self.include_sources else []
            confidence = self._calculate_confidence(retrieved_docs)

            if self.include_sources and sources:
                response_text += "\n\n##### Source documents:"
                for src in sources:
                    response_text += f"\n- [{src['title']}]({src['path']})"

            if picture_paths:
                response_text += "\n\n##### Reference images:"
                for path in picture_paths:
                    response_text += f"\n- [{path.split('/')[-1]}]({path})"

            self.logger.info(f"[LOG] Final response to user: {response_text}")
            return {"response": response_text, "sources": sources, "confidence": confidence}

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                "response": '{ "Insufficient Information", "I don\'t have enough information." }',
                "sources": [],
                "confidence": 0.0,
            }

    def _extract_sources(self, documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        sources = []
        seen = set()

        for doc in documents:
            title = doc.get("source") or doc.get("metadata", {}).get("source")
            path = doc.get("source_path") or doc.get("metadata", {}).get("source_path")
            if not title or not path:
                continue

            source_id = f"{title}|{path}"
            if source_id in seen:
                continue

            sources.append({
                "title": title,
                "path": path,
                "score": doc.get("combined_score", doc.get("rerank_score", doc.get("score", doc.get("metadata", {}).get("score", 0.0))))
            })
            seen.add(source_id)

        sources.sort(key=lambda x: x["score"], reverse=True)
        return [{"title": s["title"], "path": s["path"]} for s in sources]

    def _calculate_confidence(self, documents: List[Dict[str, Any]]) -> float:
        if not documents:
            return 0.0

        keys = ["combined_score", "rerank_score", "score"]
        for key in keys:
            if key in documents[0] or key in documents[0].get("metadata", {}):
                scores = [doc.get(key, doc.get("metadata", {}).get(key, 0)) for doc in documents[:3]]
                return sum(scores) / len(scores)

        return 0.0
