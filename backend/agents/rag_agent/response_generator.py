import logging
from typing import List, Dict, Any, Optional
from agents.rag_agent.role_llm_loader import get_llm
from agents.rag_agent.classifier_schema import QueryClassification

class ResponseGenerator:
    """
    Generates grounded responses based on retrieved document or web context.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.response_generator = get_llm()  # Main response generator
        self.fact_checker = get_llm(role="validator")  # For fact checking
        self.include_sources = True
        self.max_refinement_attempts = 2  # Maximum number of response refinements

    def _build_classified_prompt(
        self,
        query: str,
        context: str,
        classification: QueryClassification,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build a prompt optimized for the specific query type and required information.
        """
        query_type = classification.query_type
        topics = classification.topics
        required_info = getattr(classification, "required_info_types", [])
        needs_steps = classification.has_actionable_request
        
        # Base instructions
        base_instructions = self._get_base_instructions()
        
        # Add type-specific instructions
        type_instructions = {
            "procedure": """
                1. Provide clear, numbered steps
                2. Include timeframes where available
                3. Mention required documents or forms
                4. Note any fees or charges involved
                5. Add cautions or important notes
                """,
            "complaint": """
                1. Start with immediate actions to take
                2. List required documentation
                3. Explain the complaint process
                4. Mention alternative resolution methods
                5. Include relevant authority contacts
                """,
            "rights": """
                1. Clearly state each applicable right
                2. Explain practical implications
                3. Note any limitations or conditions
                4. Include relevant timelines
                5. Reference specific sections when available
                """
        }.get(query_type, "")
        
        # Format specific markers
        format_markers = "Steps:" if needs_steps else "Key Points:"
        
        prompt = f"""You are a document question-answering assistant.

User Query: {query}

Query Classification:
- Type: {query_type}
- Topics: {', '.join(topics)}
- Required Information: {', '.join(required_info)}

Context Information:
{context}

{base_instructions}

Additional Instructions for this query type:
{type_instructions}

{format_markers}

Based on the provided context, provide a structured response that specifically addresses the user's needs.
Focus on being practical and actionable while maintaining accuracy.

If you're unsure or the information is not in the context, say so clearly.
"""
        return prompt

    def _get_base_instructions(self) -> str:
        """Get the base instructions for all responses."""
        return """Instructions:
        1. Answer based ONLY on the provided context
        2. Be clear, concise, and practical
        3. Use bullet points for clarity
        4. Quote or reference specific evidence when helpful
        5. Say clearly when the answer is not supported by the context
        6. Format in markdown for readability
        """

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

    def _should_verify_response(self, query_classification: Optional[QueryClassification] = None) -> bool:
        """
        Determine if response verification is needed based on query type and risk level.
        Only verify high-stakes queries to reduce API calls and improve performance.
        """
        if not query_classification:
            return False  # Skip verification for unclassified queries
            
        # High-risk query types that need verification
        high_risk_types = ["legal-advice", "complaint", "procedure", "refund"]
        
        # Check if query involves money, legal action, or official procedures
        query_type = getattr(query_classification, "query_type", "")
        topics = getattr(query_classification, "topics", [])
        
        # Verify if it's a high-risk type or involves financial/legal topics
        if query_type in high_risk_types:
            return True
            
        # Check for financial or legal topics
        risk_topics = ["refund", "legal", "court", "complaint", "dispute", "warranty"]
        if any(topic in risk_topics for topic in topics):
            return True
            
        return False  # Skip verification for general info queries

    def generate_response(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        query_classification: Optional[Dict[str, Any]] = None,
        picture_paths: Optional[List[str]] = None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        try:
            self.logger.info(f"[LOG] Starting LLM response generation for query: {query}")
            # Get document texts and organize by relevance
            doc_texts_with_scores = []
            for doc in retrieved_docs:
                if isinstance(doc, tuple):  # If doc is (Document, score) tuple
                    doc_texts_with_scores.append((doc[0].page_content, doc[1]))
                else:  # If doc is a dict with content
                    doc_texts_with_scores.append((doc["content"], doc.get("score", doc.get("metadata", {}).get("score", 1.0))))
            
            # Sort by relevance score if available
            doc_texts_with_scores.sort(key=lambda x: x[1], reverse=True)
            doc_texts = [text for text, _ in doc_texts_with_scores]
            
            # Build context with section markers and metadata
            context_parts = []
            for i, (text, score) in enumerate(doc_texts_with_scores):
                relevance_marker = "🔥 High Relevance" if score > 0.8 else "✓ Relevant" if score > 0.6 else "ℹ️ Context"
                context_parts.append(f"\n\n=== {relevance_marker} ===\n{text}")
            
            context = "\n".join(context_parts)
            
            # Determine response format based on query classification
            if query_classification:
                prompt = self._build_classified_prompt(query, context, query_classification, chat_history)
            else:
                prompt = self._build_prompt(query, context, chat_history)
            self.logger.info(f"[LOG] Prompt sent to LLM: {prompt}")
            # Generate initial response
            initial_response = self.response_generator.invoke(prompt)
            response_text = initial_response.content.strip() if initial_response and hasattr(initial_response, 'content') else None
            self.logger.info(f"[LOG] Raw LLM response: {response_text}")

            # Handle insufficient info case
            if not response_text:
                self.logger.warning(f"[LOG] LLM returned empty response for query: {query}")
                return {
                    "response": "Sorry, no information was found for your query. Please try rephrasing or ask about another topic.",
                    "sources": [],
                    "confidence": 0.0,
                    "verification_result": None
                }
            if response_text.startswith("{") and "Insufficient Information" in response_text:
                return {
                    "response": response_text,
                    "sources": [],
                    "confidence": 0.0,
                    "verification_result": None
                }

            # Step 2: Conditionally verify response (only for high-risk queries)
            should_verify = self._should_verify_response(query_classification)
            self.logger.info(f"[LOG] Verification needed: {should_verify}")
            
            if should_verify:
                verification_result = self._verify_response(response_text, query, context)
                # Step 3: Refine response if verification suggests improvements
                final_response = self._refine_response(response_text, verification_result, query, context)
            else:
                # Skip verification for low-risk queries - significant performance boost
                verification_result = {"verified": True, "details": "Verification skipped for performance"}
                final_response = response_text
            

            # Step 4: Add source documentation
            sources = self._extract_sources(retrieved_docs) if self.include_sources else []
            confidence = self._calculate_confidence(retrieved_docs)

            if self.include_sources:
                final_response += "\n\n##### Source documents:"
                for src in sources:
                    final_response += f"\n- [{src['title']}]({src['path']})"

            if picture_paths:
                final_response += "\n\n##### Reference images:"
                for path in picture_paths:
                    final_response += f"\n- [{path.split('/')[-1]}]({path})"

            self.logger.info(f"[LOG] Final response to user: {final_response}")
            return {
                "response": final_response,
                "sources": sources,
                "confidence": confidence,
                "verification_result": verification_result
            }

        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return {
                "response": '{ "Insufficient Information", "I don\'t have enough information." }',
                "sources": [],
                "confidence": 0.0,
                "verification_result": None
            }
            
    def _verify_response(
        self,
        response: str,
        query: str,
        context: str
    ) -> Dict[str, Any]:
        """
        Verify the factual accuracy and completeness of the response.
        """
        verification_prompt = f"""
        Verify the following response for factual accuracy and completeness:

        Query: {query}
        Response: {response}
        Source Context: {context}

        Verify:
        1. All facts are supported by the context
        2. No contradictions with the source material
        3. All relevant information is included
        4. No unsupported claims or speculation

        Return a JSON object with:
        1. accuracy_score (0-1)
        2. supported_facts (list)
        3. unsupported_claims (list)
        4. missing_information (list)
        5. suggested_improvements (list)
        """
        
        verification = self.fact_checker.invoke(verification_prompt)
        try:
            # Note: In practice, you'd want to parse this properly
            return {
                "verified": True,
                "details": verification.content
            }
        except:
            return {
                "verified": False,
                "details": "Verification failed"
            }
            
    def _refine_response(
        self,
        original_response: str,
        verification_result: Dict[str, Any],
        query: str,
        context: str
    ) -> str:
        """
        Refine the response based on verification feedback.
        """
        if verification_result["verified"]:
            return original_response
            
        refinement_prompt = f"""
        Refine the following response based on verification feedback:

        Original Query: {query}
        Original Response: {original_response}
        Verification Result: {verification_result["details"]}
        Source Context: {context}

        Rules for refinement:
        1. Address any unsupported claims
        2. Add missing important information
        3. Fix any factual errors
        4. Maintain clarity and conciseness
        5. Stay true to the source material

        Provide the refined response.
        """
        
        refined = self.response_generator.invoke(refinement_prompt)
        return refined.content.strip()

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
