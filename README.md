# Document Insight Extractor

A RAG (Retrieval-Augmented Generation) tool that reads long PDF documents and produces an executive summary, key insights with page citations, and answers to your questions. It works on financial reports, legal contracts, research papers, and general text.

Built with LangChain, Groq (GPT-OSS 120B), Hugging Face embeddings, ChromaDB, FastAPI, and Streamlit.

## Features

- **Executive summary:** map-reduce summarization of the whole document, with adjustable depth.
- **Key insights:** searches the document for topics that match the chosen document type (general, financial, or legal), and lists the source pages for each point.
- **Question answering:** answers questions using only the retrieved passages, with page citations.
- **Downloadable report:** export the results as a Markdown file.

## How it works

```
PDF ──► PyPDFLoader (one document per page)
      ──► RecursiveCharacterTextSplitter (overlapping chunks)
            ├──► HuggingFaceEmbeddings ──► Chroma (vector database on disk)
            │         ├── insights: retrieve chunks per topic ──► LLM ──► points + page numbers
            │         └── Q&A:      retrieve chunks for question ──► LLM ──► cited answer
            └──► map-reduce: summarize chunks ──► merge ──► executive summary
```

Every answer is restricted to retrieved text from the document, and each point carries its page number so it can be checked against the source.

## Project structure

```
doc-insight/
├── backend/                  # FastAPI app (API endpoints)
│   └── main.py
├── frontend/                 # Streamlit app (user interface)
│   └── app.py
├── src/doc_insight/          # core AI logic
│   ├── config.py             # settings and paths
│   ├── models.py             # embedding and LLM loaders
│   ├── ingest.py             # PDF loading, chunking, indexing
│   ├── summarizer.py         # executive summary
│   ├── insights.py           # insight extraction
│   └── qa.py                 # question answering
├── data/raw/                 # uploaded PDFs (not committed)
├── outputs/                  # generated reports
├── storage/                  # vector database (not committed)
├── pyproject.toml
└── uv.lock
```

## Tech stack

| Component | Tool |
|---|---|
| LLM | Groq, `openai/gpt-oss-120b` |
| Embeddings | `BAAI/bge-small-en-v1.5` (runs locally, no API key) |
| Vector database | ChromaDB via `langchain-chroma` |
| Orchestration | LangChain |
| PDF parsing | `pypdf` |
| Backend | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Package manager | uv |

## Setup

### Prerequisites

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/) package manager
- A free Groq API key from [console.groq.com](https://console.groq.com)
- About 2 GB of free disk space for the embedding model and dependencies

### 1. Clone the repository

```bash
git clone https://github.com/roh108-glitch/doc-insight.git
cd doc-insight
```

### 2. Install dependencies

```bash
uv sync
```

This creates a `.venv` folder and installs everything from `uv.lock`.

### 3. Add your API key

Create a file named `secrets.env` in the project root:

```
GROQ_API_KEY=your_groq_key_here
```

This file is git-ignored, so your key stays on your machine.

## Running the app

The app has two parts, and each runs in its own terminal.

**Terminal 1: start the backend**

```bash
uv run uvicorn backend.main:app --reload
```

Wait for `Application startup complete`. API docs are at http://127.0.0.1:8000/docs.

**Terminal 2: start the frontend**

```bash
uv run streamlit run frontend/app.py
```

Open http://localhost:8501 in your browser.

### Using it

1. Upload a text-based PDF in the sidebar and click **Process document**. The first upload takes a minute while the embedding model loads.
2. Choose what you want: summary, insights, or question answering.
3. For insights, pick the document type (general, financial, or legal).
4. Click **Generate**, or type a question and click **Ask**.
5. Download the results as a Markdown report.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/status` | Name of the currently loaded document |
| POST | `/upload` | Upload and index a PDF |
| GET | `/summary?max_chunks=20` | Executive summary |
| GET | `/insights?document_type=general` | Key insights with page numbers |
| POST | `/ask` | Answer a question (JSON body: `{"question": "..."}`) |

## Notes and limitations

- **Text-based PDFs only.** Scanned PDFs (images with no selectable text) are rejected, because there is nothing to extract without OCR.
- **One document at a time.** Uploading a new PDF replaces the current index.
- **Rate limits.** The Groq free tier caps tokens per minute. The summary step pauses between requests to stay under the limit, so long documents take several minutes. Lower the summary depth if you hit a rate-limit error.
- **Local embeddings.** The embedding model runs on your machine, so the first run downloads it.
- **Storage is local.** Uploaded PDFs and the index live on disk and are not part of the repository.

## Possible improvements

- OCR support for scanned documents
- Per-user document storage
- Streaming responses in the UI
- Evaluation of answer accuracy against a labeled set of questions
- Deployment to Hugging Face Spaces or Render with persistent storage

## License

MIT
