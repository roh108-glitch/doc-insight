"""Model loaders. Each is created once and reused."""
from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from doc_insight.config import EMBEDDING_MODEL, GROQ_MODEL


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Turns text into vectors. Runs on your machine, no API key needed."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """Groq-hosted GPT-OSS 120B. Reads GROQ_API_KEY from the environment."""
    return ChatGroq(model=GROQ_MODEL, temperature=0.2)