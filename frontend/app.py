"""Streamlit frontend. It only talks to the FastAPI backend over HTTP."""
import os
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SECONDS = 900   # summaries of long PDFs can take several minutes
MAX_UPLOAD_MB = 50

st.set_page_config(page_title="Document Insight", page_icon="📑", layout="wide")


def call_api(method: str, path: str, **kwargs):
    """The only place that talks to the backend. Returns (data, error_message)."""
    try:
        resp = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT_SECONDS, **kwargs)
    except requests.ConnectionError:
        return None, f"Cannot reach the backend at {API_URL}. Start it first (see setup steps)."
    except requests.Timeout:
        return None, "The backend took too long to respond. Try a lower summary depth."
    except requests.RequestException as exc:
        return None, f"Request failed: {exc}"

    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except (ValueError, AttributeError):
            detail = resp.text
        if isinstance(detail, list):   # FastAPI validation errors come as a list
            detail = "; ".join(str(item.get("msg", item)) for item in detail)
        return None, str(detail)

    return resp.json(), None


def reset_results():
    st.session_state.summary = None
    st.session_state.insights = None
    st.session_state.qa_history = []


# Session state survives reruns, so results don't disappear when you click something
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
    reset_results()

# Check which document the backend has loaded
status_data, error = call_api("GET", "/status")
if error:
    st.error(error)
    st.stop()

current_doc = status_data["filename"]
if current_doc != st.session_state.doc_name:
    reset_results()                       # new document, so old results don't apply
    st.session_state.doc_name = current_doc

# ---------- Sidebar: upload ----------
with st.sidebar:
    st.header("📄 Your document")
    if current_doc:
        st.success(f"Loaded: {current_doc}")
    else:
        st.info("No document loaded yet.")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if st.button("Process document", type="primary", disabled=uploaded is None):
        if uploaded.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File is larger than {MAX_UPLOAD_MB} MB.")
        else:
            with st.spinner("Reading and indexing (the first time can take a minute)..."):
                _, error = call_api(
                    "POST", "/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                )
            if error:
                st.error(error)
            else:
                st.toast("Document is ready!")
                st.rerun()

    st.divider()
    st.caption(f"Backend: {API_URL}")

# ---------- Main page ----------
st.title("📑 Document Insight Extractor")
st.caption("Upload a report, paper, or contract, then choose what you want to see.")

if not current_doc:
    st.info("👈 Upload a PDF in the sidebar to get started.")
    st.stop()

st.subheader("What would you like?")
c1, c2, c3 = st.columns(3)
want_summary = c1.checkbox("📝 Executive summary", value=True)
want_insights = c2.checkbox("🔍 Key insights", value=True)
want_qa = c3.checkbox("💬 Question answering", value=False)

o1, o2 = st.columns(2)
doc_type = o1.selectbox(
    "Document type (for insights)",
    ["general", "financial", "legal"],
    help="Chooses which topics the insights look for.",
)
summary_depth = o2.slider(
    "Summary depth", 5, 60, 20,
    help="More sections means a more complete summary, but it takes longer.",
)

if not (want_summary or want_insights or want_qa):
    st.info("Select at least one option above.")
    st.stop()

if want_summary or want_insights:
    if st.button("Generate", type="primary"):
        if want_summary:
            with st.spinner("Writing the executive summary (this can take a few minutes)..."):
                data, error = call_api("GET", "/summary", params={"max_chunks": summary_depth})
            st.session_state.summary = {"text": data["summary"] if data else None, "error": error}

        if want_insights:
            with st.spinner("Searching the document for insights..."):
                data, error = call_api("GET", "/insights", params={"document_type": doc_type})
            st.session_state.insights = {"data": data["insights"] if data else None, "error": error}


def show_summary():
    result = st.session_state.summary
    if result is None:
        st.caption("Click Generate to create the summary.")
    elif result["error"]:
        st.error(result["error"])
    else:
        with st.container(border=True):
            st.markdown(result["text"])


def show_insights():
    result = st.session_state.insights
    if result is None:
        st.caption("Click Generate to extract insights.")
    elif result["error"]:
        st.error(result["error"])
    else:
        for category, item in result["data"].items():
            with st.container(border=True):
                st.markdown(f"**{category}**")
                st.markdown(item["answer"])
                if item["pages"]:
                    st.caption(f"Source pages: {', '.join(map(str, item['pages']))}")


def show_qa():
    question = st.text_input(
        "Ask anything about the document",
        placeholder="e.g. What are the main risks mentioned?",
        max_chars=500,
    )
    if st.button("Ask", disabled=len(question.strip()) < 3):
        with st.spinner("Searching the document..."):
            data, error = call_api("POST", "/ask", json={"question": question.strip()})
        if error:
            st.error(error)
        else:
            st.session_state.qa_history.insert(0, {"q": question.strip(), "a": data["answer"]})

    for item in st.session_state.qa_history:
        with st.container(border=True):
            st.markdown(f"**Q: {item['q']}**")
            st.markdown(item["a"])


labels = {"summary": "📝 Summary", "insights": "🔍 Insights", "qa": "💬 Q&A"}
chosen = [
    key for key, flag in
    (("summary", want_summary), ("insights", want_insights), ("qa", want_qa))
    if flag
]
for key, tab in zip(chosen, st.tabs([labels[k] for k in chosen])):
    with tab:
        if key == "summary":
            show_summary()
        elif key == "insights":
            show_insights()
        else:
            show_qa()


def build_report() -> str:
    parts = [f"# Report: {st.session_state.doc_name}", f"*Generated {datetime.now():%Y-%m-%d %H:%M}*", ""]
    summary_result = st.session_state.summary
    if summary_result and summary_result["text"]:
        parts += ["## Executive summary", "", summary_result["text"], ""]
    insights_result = st.session_state.insights
    if insights_result and insights_result["data"]:
        parts += ["## Insights", ""]
        for category, item in insights_result["data"].items():
            pages = ", ".join(map(str, item["pages"])) or "-"
            parts += [f"### {category}", "", item["answer"], "", f"*Source pages: {pages}*", ""]
    if st.session_state.qa_history:
        parts += ["## Questions and answers", ""]
        for item in st.session_state.qa_history:
            parts += [f"**Q: {item['q']}**", "", item["a"], ""]
    return "\n".join(parts)


if st.session_state.summary or st.session_state.insights or st.session_state.qa_history:
    st.divider()
    st.download_button(
        "⬇️ Download report (.md)",
        data=build_report(),
        file_name=f"{Path(current_doc).stem}_report.md",
        mime="text/markdown",
    )