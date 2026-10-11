"""Step 3: targeted insights using RAG.

Each category has a search query. We retrieve the most relevant chunks for that
query and ask the LLM to list points from those chunks only, keeping page numbers
so every point can be checked.
"""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from doc_insight.config import TOP_K
from doc_insight.ingest import format_docs, load_vector_store
from doc_insight.models import get_llm

INSIGHT_SETS = {
    "general": {
        "Key points": "main points, key findings, important facts, conclusions",
        "Important numbers and facts": "numbers, statistics, dates, names, results, data",
        "Problems and limitations": "problems, limitations, challenges, risks, weaknesses, open questions",
    },
    "financial": {
        "Key financial figures": "revenue, net income, profit margin, cash flow, growth, earnings per share",
        "Risks and uncertainties": "risks, uncertainties, litigation, threats, adverse conditions, liabilities",
        "Obligations and deadlines": "obligations, payment terms, deadlines, termination, renewal, penalties",
    },
    "legal": {
        "Key obligations": "obligations, duties, responsibilities, shall, must, required to",
        "Risks and liabilities": "liability, indemnity, penalties, breach, risks, disputes",
        "Dates and deadlines": "deadline, term, renewal, termination, notice period, due date",
    },
}

INSIGHT_PROMPT = PromptTemplate.from_template(
    "Using ONLY the context below, list up to 5 short bullet points about: {category}.\n"
    "If the context has nothing relevant, reply exactly: Nothing relevant found.\n\n"
    "CONTEXT:\n{context}\n\nBULLET POINTS:"
)


def extract_insights(store=None, document_type: str = "general") -> dict[str, dict]:
    if store is None:
        store = load_vector_store()

    queries = INSIGHT_SETS[document_type]
    retriever = store.as_retriever(search_kwargs={"k": TOP_K})
    chain = INSIGHT_PROMPT | get_llm() | StrOutputParser()

    results = {}
    for category, query in queries.items():
        docs = retriever.invoke(query)
        answer = chain.invoke({"category": category.lower(), "context": format_docs(docs)})
        pages = sorted({d.metadata.get("page", 0) + 1 for d in docs})
        results[category] = {"answer": answer, "pages": pages}
    return results