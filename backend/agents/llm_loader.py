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
    def name(self):
        return "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Loads the default sentence embedding model using HuggingFace.
    """
    settings = get_settings()
    device = os.getenv("EMBEDDING_DEVICE", "cpu")
    return NamedHuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": device},
    )
