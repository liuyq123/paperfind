"""
search.py

Semantic search and RAG query tool for paper databases.

Usage:
    paperfind search "your search query"
    paperfind search "your question" --rag
    paperfind search "your query" --source zotero
    paperfind search "your query" -k 10
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from paperfind.config import (
    CHROMA_STORE_DIR,
    ZOTERO_VECTORS_DIR,
    EMBEDDING_MODEL,
    LLM_MODEL,
)
from paperfind.search.formatting import format_document

# Configuration
ZOTERO_COLLECTION = "zotero_all"


def get_vectordb(source: str = "daily_papers") -> Chroma:
    """Get the appropriate vector database based on source."""
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if source == "zotero":
        return Chroma(
            embedding_function=embeddings,
            persist_directory=ZOTERO_VECTORS_DIR,
            collection_name=ZOTERO_COLLECTION,
        )
    else:
        return Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_STORE_DIR,
        )


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

    search_kwargs = {"k": k}
    if project_id is not None and source == "zotero":
        search_kwargs["filter"] = {"project_id": project_id}

    results = vectordb.similarity_search(query, **search_kwargs)
    return results


def search_with_scores(
    query: str,
    k: int = 5,
    source: str = "daily_papers",
) -> List[tuple]:
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
    if rag:
        print(f"\nAnswering question using RAG ({source})...\n")
        answer = rag_query(query, k=num_results, source=source)
        print(answer)
    else:
        print(f"\nSearching {source} for: {query}\n")

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
