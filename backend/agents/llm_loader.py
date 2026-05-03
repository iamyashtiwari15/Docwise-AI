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
    """
    HuggingFaceEmbeddings with optional query-time instruction.
    - BGE models: prefix the query text with an instruction string.
    - Harrier / SentenceTransformer prompt_name models: pass prompt_name to encode().
    """
    _query_instruction: str = ""
    _query_prompt_name: str = ""

    def name(self):
        return self.model_name

    def embed_query(self, text: str) -> list:
        # BGE-style: prepend instruction text before encoding
        if self._query_instruction:
            text = f"{self._query_instruction}{text}"

        # Harrier-style: use SentenceTransformer's built-in prompt_name for queries only
        if self._query_prompt_name:
            embedding = self._client.encode(
                [text],
                normalize_embeddings=True,
                prompt_name=self._query_prompt_name,
            )
            return embedding[0].tolist()

        return super().embed_query(text)


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Loads the sentence embedding model.
    - Harrier (Microsoft): decoder-only, uses prompt_name="web_search_query" for queries.
    - BGE: query instruction prefix injected at embed_query time.
    """
    settings = get_settings()
    model_name = settings.embedding_model_name
    device = settings.embedding_device

    model_kwargs: dict = {"device": device}

    model = NamedHuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True},
    )

    if "harrier" in model_name.lower():
        model._query_prompt_name = "web_search_query"
    elif "bge" in model_name.lower():
        model._query_instruction = "Represent this sentence for searching relevant passages: "

    return model
