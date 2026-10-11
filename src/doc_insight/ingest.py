"""Step 1: load a PDF, split it into chunks, and store their embeddings in Chroma."""
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from doc_insight.config import CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, STORAGE_DIR
from doc_insight.models import get_embeddings


def load_pdf(pdf_path: Path) -> list[Document]:
    """Returns one Document per page. Metadata includes the page number (0-based)."""
    return PyPDFLoader(str(pdf_path)).load()


def split_pages(pages: list[Document]) -> list[Document]:
    """Breaks pages into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(pages)


def build_vector_store(pdf_path: Path) -> Chroma:
    """Indexes one PDF, replacing any previous document's chunks."""
    chunks = split_pages(load_pdf(pdf_path))
    if not chunks:
        raise ValueError("No readable text found. Scanned PDFs (images only) are not supported.")

    store = Chroma(
        persist_directory=str(STORAGE_DIR),
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )
    try:
        store.delete_collection()   # clear the previous document's chunks
    except Exception:
        pass                        # no previous collection exists yet

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(STORAGE_DIR),
        collection_name=COLLECTION_NAME,
    )


def load_vector_store() -> Chroma:
    """Reopens the saved index without re-processing the PDF."""
    if not STORAGE_DIR.exists():
        raise FileNotFoundError("No index found. Upload a document first.")
    return Chroma(
        persist_directory=str(STORAGE_DIR),
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def format_docs(docs: list[Document]) -> str:
    """Joins retrieved chunks and tags each with its page, so the LLM can cite it."""
    return "\n\n".join(
        f"[page {d.metadata.get('page', 0) + 1}] {d.page_content}" for d in docs
    )