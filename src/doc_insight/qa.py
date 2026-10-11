"""Step 4: answer questions about the document with a retrieval-augmented chain."""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from doc_insight.config import TOP_K
from doc_insight.ingest import format_docs, load_vector_store
from doc_insight.models import get_llm

QA_PROMPT = PromptTemplate.from_template(
    "Answer the question using ONLY the context. Cite page numbers like [page 4]. "
    'If the answer is not in the context, say "I could not find that in the document."\n\n'
    "CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
)


def build_qa_chain(store=None):
    if store is None:
        store = load_vector_store()
    retriever = store.as_retriever(search_kwargs={"k": TOP_K})

    # question -> retrieve chunks -> format them -> prompt -> LLM -> text
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | QA_PROMPT
        | get_llm()
        | StrOutputParser()
    )


def answer_question(question: str) -> str:
    return build_qa_chain().invoke(question)