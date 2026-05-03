
# agents/llm_loader.py

import os
from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from dotenv import load_dotenv

from agents.llm_loader import get_embedding_model as get_shared_embedding_model
from agents.llm_loader import get_llm as get_base_llm

load_dotenv()

@lru_cache(maxsize=8)
def get_llm(role: str = "default"):
    """
    Loads the LLM with specific system prompts based on role.
    
    Args:
        role: Role for the LLM (default, planner, validator, etc.)
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")
    
    if not groq_api_key:
        raise ValueError("❌ GROQ_API_KEY not found in .env file.")
    
    system_prompts = {
        "default": "You are a helpful assistant.",
        "rag_answerer": (
            "You are an expert teacher and document scholar. Your job is to read the retrieved document passages "
            "and explain concepts to the user the way a knowledgeable, engaging teacher would — clearly, "
            "thoroughly, and with genuine depth of understanding.\n\n"
            "Teaching principles you must follow:\n"
            "1. GROUND every explanation in the retrieved <context> — use only facts from the documents; do not add outside knowledge.\n"
            "2. EXPLAIN, don't just list — weave facts into a coherent, well-reasoned narrative. Help the user truly understand, not just skim a summary.\n"
            "3. STRUCTURE your answer like a lesson:\n"
            "   • Open with a brief framing sentence that sets up the topic and its significance.\n"
            "   • Develop the explanation using ## markdown headings for each distinct aspect or argument.\n"
            "   • Under each heading, explain in clear prose supported by specific evidence from the context.\n"
            "   • Use bullet points for lists of conditions, steps, arguments, or exceptions.\n"
            "   • End with a concise ## Key Takeaways section.\n"
            "4. PRESERVE DETAIL — include every argument, counter-argument, date, number, condition, threshold, "
            "and exception found in the context. Never flatten or omit nuance.\n"
            "5. REPRODUCE TABLES as markdown tables — never convert tabular data to prose.\n"
            "6. CITE PAGES inline where passages carry page numbers — write '(Page N)' after the relevant sentence.\n"
            "7. When context is insufficient, reply with exactly: INSUFFICIENT_INFORMATION\n"
            "8. PREFER DEPTH — a thorough, well-taught answer is always better than a brief one."
        ),
        "planner": (
            "You are a query planning expert. Analyze user queries to:\n"
            "1. Identify key concepts and relationships\n"
            "2. Break down complex queries into sub-queries\n"
            "3. Determine required information types\n"
            "4. Plan the retrieval strategy"
        ),
        "validator": (
            "You are a fact-checking expert. Your role is to:\n"
            "1. Verify response accuracy against source documents\n"
            "2. Identify unsupported claims\n"
            "3. Ensure response completeness\n"
            "4. Suggest improvements"
        ),
        "expander": (
            "You are a query expansion expert. Your role is to:\n"
            "1. Identify key concepts in user queries\n"
            "2. Add relevant domain-specific terminology\n"
            "3. Include synonyms and related terms\n"
            "4. Maintain query intent and context"
        ),
        "multi_query": (
            "You are a retrieval query optimizer. Produce a few short search variants that preserve the user's meaning while improving retrieval recall.\n"
            "1. Keep each variant focused and concrete\n"
            "2. Split compound requests into separate targeted search queries when helpful\n"
            "3. Do not invent facts or add new requirements\n"
            "4. Return only retrieval-friendly variants, not final answers"
        ),
        "hyde": (
            "You are a document passage writer. Given a user question, write a single dense factual passage "
            "that would appear in an authoritative document and directly answer the question. "
            "Write only the passage — no preamble, no hedging, no 'based on the question'. "
            "Include specific numbers, dates, conditions, thresholds and procedures that a real document would contain. "
            "Write as if you are the document itself."
        ),
        "query_rewriter": (
            "You are a query rewriter. Convert follow-up questions into fully self-contained questions by resolving any references to prior conversation. "
            "Do not change the meaning. Return only the rewritten question, nothing else."
        ),
    }
    
    llm = get_base_llm(model_name=model_name)
    
    # Create a wrapped version that includes the system prompt in each call
    system_prompt = system_prompts.get(role, system_prompts["default"])
    
    class WrappedLLM:
        def __init__(self, base_llm, system_prompt):
            self.base_llm = base_llm
            self.system_prompt = system_prompt
            
        def invoke(self, prompt: str):
            # Create a chat template that includes the system prompt
            chat_template = ChatPromptTemplate.from_messages([
                SystemMessage(content=self.system_prompt),
                HumanMessagePromptTemplate.from_template("{input}")
            ])
            
            # Format the messages with the actual prompt
            messages = chat_template.format_messages(input=prompt)
            
            # Invoke the base LLM with the formatted messages
            return self.base_llm.invoke(messages)
    
    return WrappedLLM(llm, system_prompt)
class NamedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    def name(self):
        return "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding_model():
    """
    Loads the default sentence embedding model using HuggingFace.
    """
    return get_shared_embedding_model()
