"""Vector store helpers for paper fetchers."""

import sqlite3
import time
from pathlib import Path

from paperfind.config import DAILY_PAPERS_DB, get_chroma_store_dir
from paperfind.embeddings import get_embeddings

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
        from langchain_chroma import Chroma
    except ImportError:
        print("    Error: langchain-chroma not installed")
        print("    Run: pip install langchain-chroma")
        return 0

    print("\n[Vectors] Updating vector embeddings for new papers...")

    conn = sqlite3.connect(DAILY_PAPERS_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    total_docs = 0
    chroma_dir = get_chroma_store_dir()
    embeddings = get_embeddings()
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=chroma_dir,
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
    print(f"    Upserted {total_docs} documents into {chroma_dir}/")
    return total_docs


def rebuild_vectors() -> None:
    """Rebuild the vector database from SQLite."""
    import shutil

    print("\n[Vectors] Rebuilding vector embeddings...")

    try:
        from langchain_chroma import Chroma
    except ImportError:
        print("    Error: langchain-chroma not installed")
        print("    Run: pip install langchain-chroma")
        return

    # Check if database exists before proceeding
    if not Path(DAILY_PAPERS_DB).exists():
        print("    Error: No papers database found. Run 'paperfind fetch' first.")
        return

    conn = sqlite3.connect(DAILY_PAPERS_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Check if works table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='works'")
    if not cur.fetchone():
        conn.close()
        print("    Error: No papers in database. Run 'paperfind fetch' first.")
        return

    # Now safe to clear existing vector store
    chroma_dir = get_chroma_store_dir()
    if Path(chroma_dir).exists():
        shutil.rmtree(chroma_dir)
        print(f"    Cleared existing store at {chroma_dir}/")

    cur.execute("SELECT doi, title, authors, abstract, created_date, type, source FROM works")

    docs = _build_documents(cur.fetchall())

    conn.close()

    if not docs:
        print("    No documents to embed")
        return

    print(f"    Embedding {len(docs)} documents...")

    embeddings = get_embeddings()

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
                        persist_directory=chroma_dir,
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

    print(f"    Done! Vector store saved to {chroma_dir}/")
