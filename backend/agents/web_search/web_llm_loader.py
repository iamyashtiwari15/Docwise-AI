# agents/llm_loader.py

import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    """
    Loads the default LLM (Groq-based Mixtral/LLama3) using the Groq API key from the .env file.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("LLM_MODEL_NAME", "llama-3.3-70b-versatile")  # or llama3-70b-8192
    
    if not groq_api_key:
        raise ValueError("❌ GROQ_API_KEY not found in .env file.")
    
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name
    )
    return llm

class NamedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    def name(self):
        return "sentence-transformers/all-MiniLM-L6-v2"

def get_embedding_model():
    """
    Loads the default sentence embedding model using HuggingFace.
    """
    # model_name = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    
    # embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # return embedding
    return NamedHuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")