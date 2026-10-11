"""Step 2: executive summary using map-reduce.

Map:    summarize each chunk separately.
Reduce: merge those summaries, repeating if needed, until they fit in one prompt.
"""
from pathlib import Path
import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from doc_insight.ingest import load_pdf, split_pages
from doc_insight.models import get_llm

CHUNK_PROMPT = PromptTemplate.from_template(
    "Summarize the text below in 3 short bullet points.\n\nTEXT:\n{text}\n\nBULLET POINTS:"
)
FINAL_PROMPT = PromptTemplate.from_template(
    "Below are notes from different parts of a document."
    "Write an executive summary of 8 bullet points covering the main topics, key numbers, and conclusions.\n\n"
    "NOTES:\n{text}\n\nEXECUTIVE SUMMARY:"
)
MAX_REDUCE_CHARS = 6000   # keeps each prompt within the local model's input limit


def _sample_evenly(items: list, n: int | None) -> list:
    """Picks n items spread across the whole document, so the summary covers all of it."""
    if n is None or len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def _group(texts: list[str], max_chars: int) -> list[str]:
    """Packs short texts into groups of at most max_chars."""
    groups, current = [], ""
    for text in texts:
        if current and len(current) + len(text) > max_chars:
            groups.append(current)
            current = ""
        current += text + "\n"
    if current:
        groups.append(current)
    return groups

def _run_slowly(chain, inputs: list[dict]) -> list[str]:
    """Runs the chain one input at a time, with a pause, to stay under Groq's rate limit."""
    results = []
    for item in inputs:
        results.append(chain.invoke(item))
        time.sleep(2)   # pause between calls
    return results

def summarize_document(pdf_path: Path, max_chunks: int | None = 60) -> str:
    chunks = _sample_evenly(split_pages(load_pdf(pdf_path)), max_chunks)
    llm = get_llm()
    chunk_chain = CHUNK_PROMPT | llm | StrOutputParser()

    # Map: summarize each chunk (one at a time, to avoid rate limits)
    notes = _run_slowly(chunk_chain, [{"text": c.page_content} for c in chunks])

    # Reduce: merge notes until they fit in a single prompt
    while len("\n".join(notes)) > MAX_REDUCE_CHARS:
        groups = _group(notes, MAX_REDUCE_CHARS)
        if len(groups) >= len(notes):   # safety stop: merging isn't shrinking anything
            break
        notes = _run_slowly(chunk_chain, [{"text": g} for g in groups])

    final_chain = FINAL_PROMPT | llm | StrOutputParser()
    return final_chain.invoke({"text": "\n".join(notes)})