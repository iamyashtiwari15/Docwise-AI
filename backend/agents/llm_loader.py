# agents/llm_loader.py

import os
from functools import lru_cache

from langchain_groq import ChatGroq

from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from core.config import get_settings

load_dotenv()


@lru_cache(maxsize=4)
def get_llm(model_name: str | None = None):
    """
    Loads the default LLM (Groq-based Mixtral/LLama3) using the Groq API key from the .env file.
    """
    settings = get_settings()
    groq_api_key = os.getenv("GROQ_API_KEY")
    resolved_model_name = model_name or settings.llm_model_name
    
    if not groq_api_key:
        raise ValueError("❌ GROQ_API_KEY not found in .env file.")
    
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=resolved_model_name,
        temperature=0,
    )
    return llm

class NamedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """HuggingFaceEmbeddings with an optional query-time instruction prefix (for BGE models)."""
    _query_instruction: str = ""

    def name(self):
        return self.model_name

    def embed_query(self, text: str) -> list:
        prefixed = f"{self._query_instruction}{text}" if self._query_instruction else text
        return super().embed_query(prefixed)


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Loads the sentence embedding model. For BGE-family models the query instruction
    is injected at embed_query time (not as a constructor arg — Pydantic forbids extras).
    """
    settings = get_settings()
    model_name = settings.embedding_model_name
    device = settings.embedding_device

    model = NamedHuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )

    # BGE models require a query-side instruction prefix for asymmetric retrieval quality
    if "bge" in model_name.lower():
        model._query_instruction = "Represent this sentence for searching relevant passages: "

    return model
