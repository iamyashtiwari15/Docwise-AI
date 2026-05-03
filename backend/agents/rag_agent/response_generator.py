import logging
from typing import List, Dict, Any, Optional
from agents.rag_agent.role_llm_loader import get_llm


class ResponseGenerator:
    """
    Generates grounded responses based on retrieved document or web context.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.llm = get_llm(role="rag_answerer")
        self.include_sources = True

    def _format_context(self, doc_texts_with_scores: List[tuple]) -> str:
        parts = []
        for i, item in enumerate(doc_texts_with_scores, start=1):
            if len(item) == 3:
                text, score, metadata = item
            else:
                text, score = item
                metadata = {}
            label = "primary" if score > 0.75 else "supporting" if score > 0.45 else "background"
            page = metadata.get("page_number")
            page_attr = f' page="{page}"' if page else ""
            source = metadata.get("source", "")
            source_attr = f' source="{source}"' if source else ""
            parts.append(f'<passage id="{i}" relevance="{label}"{page_attr}{source_attr}>\n{text}\n</passage>')
        return "\n\n".join(parts)

    def _build_prompt(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        history_section = ""
        if chat_history:
            turns = "\n".join(
                f"{msg['role'].capitalize()}: {msg['content']}"
                for msg in chat_history
            )
            history_section = f"<conversation_history>\n{turns}\n</conversation_history>\n\n"

        return (
            f"{history_section}"
            f"<context>\n{context}\n</context>\n\n"
            f"<question>\n{query}\n</question>\n\n"
            "Think through this as a teacher preparing a lesson on the question above:\n\n"
            "Step 1 — Identify the core concept the user wants to understand and every sub-aspect implied by the question.\n"
            "Step 2 — Survey ALL passages in <context>. Gather every relevant argument, counter-argument, "
            "definition, example, number, date, condition, threshold, and exception.\n"
            "Step 3 — Compose a well-structured teaching answer:\n"
            "  • Open with a clear framing paragraph: what the topic is and why it matters.\n"
            "  • Use ## markdown headings to separate each distinct aspect, argument, or sub-topic.\n"
            "  • Under each heading, write explanatory prose — connect the evidence into reasoning, "
            "not just a bullet dump. Teach the concept, don't just quote the document.\n"
            "  • Use bullet points (−) for lists of conditions, steps, arguments, or named items.\n"
            "  • If a passage contains a table relevant to the question, reproduce it as a markdown table.\n"
            "  • Cite page numbers inline wherever the passage has a page attribute — e.g. '(Page 5)'.\n"
            "Step 4 — Close with a ## Key Takeaways section: 3–6 bullet points summarising the most "
            "important points the user should walk away with.\n\n"
            "If no relevant information is present in the context, respond with exactly: INSUFFICIENT_INFORMATION"
        )

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
                    doc_texts_with_scores.append((doc[0].page_content, doc[1], {}))
                else:
                    ranking_score = doc.get(
                        "combined_score",
                        doc.get(
                            "rerank_score",
                            doc.get("score", doc.get("metadata", {}).get("score", 1.0)),
                        ),
                    )
                    doc_texts_with_scores.append((doc["content"], ranking_score, doc.get("metadata", {})))

            doc_texts_with_scores.sort(key=lambda x: x[1], reverse=True)

            context = self._format_context(doc_texts_with_scores)
            prompt = self._build_prompt(query, context, chat_history)
            self.logger.debug("[LOG] Prompt sent to LLM: %s", prompt)

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
            # Detect the sentinel token (or any LLM paraphrase of it)
            if "INSUFFICIENT_INFORMATION" in response_text or (
                "Insufficient Information" in response_text and len(response_text) < 200
            ):
                self.logger.warning("[LOG] LLM signalled insufficient context for query: %s", query)
                return {
                    "response": "I don't have enough information in the provided context to answer that question.",
                    "sources": [],
                    "confidence": 0.0,
                }

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
                "response": "Sorry, something went wrong while generating the response. Please try again.",
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
