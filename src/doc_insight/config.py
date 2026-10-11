"""All paths and settings live here."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / "secrets.env")

DATA_DIR = ROOT_DIR / "data" / "raw"
STORAGE_DIR = ROOT_DIR / "storage" / "chroma"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Chunking (characters)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K = 4
COLLECTION_NAME = "documents"

# Embeddings: free, runs locally, no API key
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM: Groq GPT-OSS 120B
GROQ_MODEL = "openai/gpt-oss-120b"
