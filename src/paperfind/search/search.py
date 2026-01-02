"""
search.py

Semantic search and RAG query tool for paper databases.

Usage:
    paperfind search "your search query"
    paperfind search "your question" --rag
    paperfind search "your query" --source zotero
    paperfind search "your query" -k 10
"""

from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from paperfind.config import LLM_MODEL
from paperfind.logging import get_logger
from paperfind.search.formatting import format_document
from paperfind.search.utils import check_vector_store, warn_if_empty
from paperfind.vectorstore import get_vector_store

logger = get_logger(__name__)

def get_vectordb(source: str = "daily_papers"):
    """Get the appropriate vector database based on source."""
    return get_vector_store(source)


def search(
    query: str,
    k: int = 5,
    source: str = "daily_papers",
    project_id: Optional[int] = None,
) -> List[Document]:
    """
    Perform semantic search on the paper database.

    Args:
        query: Search query string
        k: Number of results to return
        source: "daily_papers" or "zotero"
        project_id: Filter by project ID (zotero only)

    Returns:
        List of matching documents
    """
    vectordb = get_vectordb(source)
    warn_if_empty(vectordb, source)

    search_kwargs = {"k": k}
    if project_id is not None and source == "zotero":
        search_kwargs["filter"] = {"project_id": project_id}

    results = vectordb.similarity_search(query, **search_kwargs)
    return results


def search_with_scores(
    query: str,
    k: int = 5,
    source: str = "daily_papers",
) -> List[Tuple[Document, float]]:
    """
    Perform semantic search and return results with similarity scores.

    Args:
        query: Search query string
        k: Number of results to return
        source: "daily_papers" or "zotero"

    Returns:
        List of (document, score) tuples
    """
    vectordb = get_vectordb(source)
    warn_if_empty(vectordb, source)
    results = vectordb.similarity_search_with_score(query, k=k)
    return results


def rag_query(
    question: str,
    k: int = 5,
    source: str = "daily_papers",
) -> str:
    """
    Answer a question using RAG (Retrieval-Augmented Generation).

    Args:
        question: Question to answer
        k: Number of documents to retrieve for context
        source: "daily_papers" or "zotero"

    Returns:
        Generated answer string
    """
    vectordb = get_vectordb(source)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant for scientific paper discovery.\n"
            "Use the provided context (titles and abstracts) to answer the question.\n"
            "If you are not sure, say you don't know.\n\n"
            "Context:\n{context}",
        ),
        ("human", "{question}"),
    ])

    format_docs = RunnableLambda(
        lambda docs: "\n\n".join(d.page_content for d in docs)
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)


def run_search(
    query: str,
    num_results: int = 5,
    source: str = "daily_papers",
    rag: bool = False,
    scores: bool = False,
    project_id: Optional[int] = None,
) -> None:
    """Run semantic search with parsed parameters."""
    if not check_vector_store(source):
        return

    if rag:
        logger.info(f"Answering question using RAG ({source})...")
        answer = rag_query(query, k=num_results, source=source)
        print(answer)
    else:
        logger.info(f"Searching {source} for: {query}")

        if scores:
            results = search_with_scores(query, k=num_results, source=source)
            for i, (doc, score) in enumerate(results):
                print(format_document(doc, rank=i + 1, score=score))
        else:
            results = search(
                query,
                k=num_results,
                source=source,
                project_id=project_id,
            )
            for i, doc in enumerate(results):
                print(format_document(doc, rank=i + 1))

    print()
