"""
search.py

Semantic search and RAG query tool for paper databases.

Usage:
    paperfind search "your search query"
    paperfind search "your question" --rag
    paperfind search "your query" --source zotero
    paperfind search "your query" -k 10
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.vectorstores.base import VectorStore
from langchain_openai import ChatOpenAI

from paperfind.config import LLM_MODEL
from paperfind.db import ZOTERO_SCHEMA, get_conn, placeholder, qualify_table
from paperfind.logging import get_logger
from paperfind.search.formatting import format_document
from paperfind.search.utils import check_vector_store, warn_if_empty
from paperfind.vectorstore import get_vector_store

logger = get_logger(__name__)


def get_collection_zotero_keys(collection_name: str) -> Set[str]:
    """Get all zotero_keys in a collection."""
    conn = get_conn(ZOTERO_SCHEMA)
    cur = conn.cursor()
    collections_table = qualify_table(ZOTERO_SCHEMA, "collections")
    items_table = qualify_table(ZOTERO_SCHEMA, "items")
    item_collections_table = qualify_table(ZOTERO_SCHEMA, "item_collections")
    ph = placeholder()

    # Find collection ID by name or key
    cur.execute(
        f"SELECT id FROM {collections_table} WHERE collection_key = {ph} OR LOWER(name) = LOWER({ph})",
        (collection_name, collection_name)
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return set()

    collection_id = row["id"]

    # Get zotero_keys in collection
    cur.execute(
        f"""
        SELECT i.zotero_key
        FROM {items_table} i
        JOIN {item_collections_table} ic ON i.id = ic.item_id
        WHERE ic.collection_id = {ph}
        """,
        (collection_id,)
    )
    keys = {r["zotero_key"] for r in cur.fetchall()}
    conn.close()
    return keys

def get_vectordb(source: str = "daily_papers") -> VectorStore:
    """Get the appropriate vector database based on source."""
    return get_vector_store(source)


def search(
    query: str,
    k: int = 5,
    source: str = "daily_papers",
    collection: Optional[str] = None,
) -> List[Document]:
    """
    Perform semantic search on the paper database.

    Args:
        query: Search query string
        k: Number of results to return
        source: "daily_papers" or "zotero"
        collection: Filter by Zotero collection (zotero source only)

    Returns:
        List of matching documents
    """
    vectordb = get_vectordb(source)
    warn_if_empty(vectordb, source)

    # If filtering by collection, get allowed zotero_keys
    allowed_keys: Optional[Set[str]] = None
    if collection and source == "zotero":
        allowed_keys = get_collection_zotero_keys(collection)
        if not allowed_keys:
            logger.warning(f"Collection '{collection}' not found or empty.")
            return []

    # Search with more results if filtering
    search_k = k * 3 if allowed_keys else k
    results = vectordb.similarity_search(query, k=search_k)

    # Filter by collection if needed
    if allowed_keys:
        results = [
            doc for doc in results
            if doc.metadata.get("zotero_key") in allowed_keys
        ][:k]

    return results


def search_with_scores(
    query: str,
    k: int = 5,
    source: str = "daily_papers",
    collection: Optional[str] = None,
) -> List[Tuple[Document, float]]:
    """
    Perform semantic search and return results with similarity scores.

    Args:
        query: Search query string
        k: Number of results to return
        source: "daily_papers" or "zotero"
        collection: Filter by Zotero collection (zotero source only)

    Returns:
        List of (document, score) tuples
    """
    vectordb = get_vectordb(source)
    warn_if_empty(vectordb, source)

    allowed_keys: Optional[Set[str]] = None
    if collection and source == "zotero":
        allowed_keys = get_collection_zotero_keys(collection)
        if not allowed_keys:
            logger.warning(f"Collection '{collection}' not found or empty.")
            return []

    search_k = k * 3 if allowed_keys else k
    results = vectordb.similarity_search_with_score(query, k=search_k)

    if allowed_keys:
        results = [
            (doc, score)
            for doc, score in results
            if doc.metadata.get("zotero_key") in allowed_keys
        ][:k]

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
    collection: Optional[str] = None,
) -> None:
    """Run semantic search with parsed parameters."""
    if not check_vector_store(source):
        return

    if rag:
        logger.info(f"Answering question using RAG ({source})...")
        answer = rag_query(query, k=num_results, source=source)
        logger.info(answer)
    else:
        logger.info(f"Searching {source} for: {query}")

        if scores:
            results = search_with_scores(
                query,
                k=num_results,
                source=source,
                collection=collection,
            )
            for i, (doc, score) in enumerate(results):
                logger.info(format_document(doc, rank=i + 1, score=score))
        else:
            results = search(
                query,
                k=num_results,
                source=source,
                collection=collection,
            )
            for i, doc in enumerate(results):
                logger.info(format_document(doc, rank=i + 1))

    logger.info("")
