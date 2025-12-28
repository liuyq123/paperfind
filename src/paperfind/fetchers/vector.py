"""Vector store helpers for paper fetchers."""

import sqlite3
import time

from paperfind.config import DAILY_PAPERS_DB, CHROMA_STORE_DIR, EMBEDDING_MODEL

DEFAULT_BATCH_SIZE = 100


def _build_documents(rows: list[sqlite3.Row]) -> list:
    docs = []
    try:
        from langchain_core.documents import Document
    except ImportError:
        return docs

    for row in rows:
        title = (row["title"] or "")[:1000]
        abstract = (row["abstract"] or "")[:4000]
        content = title
        if abstract:
            content += "\n\n" + abstract

        metadata = {
            "doi": row["doi"],
            "authors": row["authors"],
            "created_date": row["created_date"],
            "type": row["type"],
            "source": row["source"],
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def upsert_vectors_for_dois(dois: list[str], batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    """Upsert vector embeddings for specific DOIs."""
    if not dois:
        return 0

    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
    except ImportError:
        print("    Error: langchain packages not installed")
        print("    Run: pip install langchain-openai langchain-chroma")
        return 0

    print("\n[Vectors] Updating vector embeddings for new papers...")

    conn = sqlite3.connect(DAILY_PAPERS_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    total_docs = 0
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, chunk_size=50)
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_STORE_DIR,
    )

    for i in range(0, len(dois), batch_size):
        chunk = dois[i:i + batch_size]
        placeholders = ",".join("?" for _ in chunk)
        cur.execute(
            f"""
            SELECT doi, title, authors, abstract, created_date, type, source
            FROM works
            WHERE doi IN ({placeholders})
            """,
            chunk,
        )
        rows = cur.fetchall()
        docs = _build_documents(rows)
        if not docs:
            continue

        ids = [doc.metadata["doi"] for doc in docs]
        vectordb.add_documents(docs, ids=ids)
        total_docs += len(docs)

    conn.close()
    print(f"    Upserted {total_docs} documents into {CHROMA_STORE_DIR}/")
    return total_docs


def rebuild_vectors() -> None:
    """Rebuild the vector database from SQLite."""
    print("\n[Vectors] Rebuilding vector embeddings...")

    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
    except ImportError:
        print("    Error: langchain packages not installed")
        print("    Run: pip install langchain-openai langchain-chroma")
        return

    conn = sqlite3.connect(DAILY_PAPERS_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT doi, title, authors, abstract, created_date, type, source FROM works")

    docs = _build_documents(cur.fetchall())

    conn.close()

    if not docs:
        print("    No documents to embed")
        return

    print(f"    Embedding {len(docs)} documents...")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, chunk_size=50)

    batch_size = DEFAULT_BATCH_SIZE
    vectordb = None

    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        print(
            f"    Batch {i//batch_size + 1}/{(len(docs)-1)//batch_size + 1} ({len(batch)} docs)"
        )

        max_retries = 5
        for attempt in range(max_retries):
            try:
                if vectordb is None:
                    vectordb = Chroma.from_documents(
                        documents=batch,
                        embedding=embeddings,
                        persist_directory=CHROMA_STORE_DIR,
                    )
                else:
                    vectordb.add_documents(batch)
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait_time = (2 ** attempt) + 1
                    print(f"    Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise
        else:
            print(f"    Failed after {max_retries} retries, skipping batch")

    print(f"    Done! Vector store saved to {CHROMA_STORE_DIR}/")
