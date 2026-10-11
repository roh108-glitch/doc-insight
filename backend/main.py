"""FastAPI backend. The Streamlit frontend talks to these endpoints over HTTP.

  GET  /status    which document is currently loaded
  POST /upload    upload a PDF and index it
  GET  /summary   executive summary of the loaded document
  GET  /insights  key insights (choose a document type)
  POST /ask       answer one question about the loaded document

Interactive docs: http://127.0.0.1:8000/docs
"""
import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from doc_insight.config import DATA_DIR, ROOT_DIR
from doc_insight.ingest import build_vector_store, load_vector_store
from doc_insight.insights import extract_insights
from doc_insight.qa import answer_question
from doc_insight.summarizer import summarize_document

MAX_UPLOAD_MB = 50
STATE_FILE = ROOT_DIR / "storage" / "current_doc.json"

app = FastAPI(title="Document Insight API", version="1.0")


class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


def _read_state() -> str | None:
    """Returns the loaded document's filename, or None if nothing is loaded."""
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8")).get("filename")
    except (ValueError, OSError, AttributeError):
        return None


def _require_document() -> Path:
    """Stops with a clear message if no document has been uploaded."""
    name = _read_state()
    if not name or not (DATA_DIR / name).exists():
        raise HTTPException(status_code=409, detail="No document loaded. Upload a PDF first.")
    return DATA_DIR / name


def _run_task(task, *args, **kwargs):
    """Runs a model or indexing task. Unexpected failures become a readable API error."""
    try:
        return task(*args, **kwargs)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Processing failed: {exc}") from exc


@app.get("/status")
def status():
    return {"filename": _read_state()}


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    limit = MAX_UPLOAD_MB * 1024 * 1024
    content = file.file.read(limit + 1)
    if len(content) > limit:
        raise HTTPException(status_code=413, detail=f"File is larger than {MAX_UPLOAD_MB} MB.")
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="The file is not a valid PDF.")

    safe_name = Path(file.filename).name
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = DATA_DIR / safe_name
    pdf_path.write_bytes(content)

    # Nothing counts as loaded until indexing succeeds
    STATE_FILE.unlink(missing_ok=True)
    _run_task(build_vector_store, pdf_path)

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps({"filename": safe_name}), encoding="utf-8")
    return {"filename": safe_name, "message": "Document indexed and ready."}


@app.get("/summary")
def summary(max_chunks: int = Query(20, ge=5, le=60)):
    pdf_path = _require_document()
    text = _run_task(summarize_document, pdf_path, max_chunks=max_chunks)
    return {"summary": text}


@app.get("/insights")
def insights(document_type: Literal["general", "financial", "legal"] = "general"):
    _require_document()
    store = _run_task(load_vector_store)
    return {"insights": _run_task(extract_insights, store, document_type=document_type)}


@app.post("/ask")
def ask(body: Question):
    _require_document()
    return {"answer": _run_task(answer_question, body.question)}