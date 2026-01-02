"""Vector store helpers for paper fetchers."""
import time
from pathlib import Path
from typing import Any, List, Mapping, Sequence

from paperfind.config import DAILY_PAPERS_DB, get_chroma_store_dir
from paperfind.db import (
    DAILY_SCHEMA,
    get_conn,
    is_postgres,
    placeholders,
    qualify_table,
    table_exists,
)
from paperfind.logging import get_logger
from paperfind.vectorstore import get_vector_store, get_vector_store_backend

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 100


def _build_documents(rows: Sequence[Mapping[str, Any]]) -> List[Any]:
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


def upsert_vectors_for_dois(dois: List[str], batch_size: int = DEFAULT_BATCH_SIZE) -> int:
    """Upsert vector embeddings for specific DOIs."""
    if not dois:
        return 0

    logger.info("[Vectors] Updating vector embeddings for new papers...")

    try:
        vectordb = get_vector_store()
    except (ImportError, ValueError) as exc:
        logger.error(str(exc))
        return 0

    conn = get_conn(DAILY_SCHEMA)
    cur = conn.cursor()

    total_docs = 0

    for i in range(0, len(dois), batch_size):
        chunk = dois[i : i + batch_size]
        placeholders_sql = placeholders(len(chunk))
        table = qualify_table(DAILY_SCHEMA, "works")
        cur.execute(
            f"""
            SELECT doi, title, authors, abstract, created_date, type, source
            FROM {table}
            WHERE doi IN ({placeholders_sql})
            """,
            chunk,
        )
        rows = cur.fetchall()
        docs = _build_documents(rows)
        if not docs:
            continue

        ids = [str(doc.metadata["doi"]) for doc in docs]
        vectordb.add_documents(docs, ids=ids)
        total_docs += len(docs)

    conn.close()
    logger.info(f"    Upserted {total_docs} documents into vector store")
    return total_docs


def rebuild_vectors() -> None:
    """Rebuild the vector database from the papers database."""
    import shutil

    logger.info("[Vectors] Rebuilding vector embeddings...")

    # Check if database exists before proceeding (SQLite only)
    if not is_postgres() and not Path(DAILY_PAPERS_DB).exists():
        logger.error("No papers database found. Run 'paperfind fetch' first.")
        return

    try:
        conn = get_conn(DAILY_SCHEMA)
    except Exception as exc:
        logger.error(f"Failed to connect to database: {exc}")
        return

    cur = conn.cursor()

    if not table_exists(conn, DAILY_SCHEMA, "works"):
        conn.close()
        logger.error("No papers in database. Run 'paperfind fetch' first.")
        return

    backend = get_vector_store_backend()
    if backend == "chroma":
        chroma_dir = get_chroma_store_dir()
        if Path(chroma_dir).exists():
            shutil.rmtree(chroma_dir)
            logger.debug(f"Cleared existing store at {chroma_dir}/")

    table = qualify_table(DAILY_SCHEMA, "works")
    cur.execute(
        f"SELECT doi, title, authors, abstract, created_date, type, source FROM {table}"
    )

    docs = _build_documents(cur.fetchall())

    conn.close()

    if not docs:
        logger.warning("No documents to embed")
        return

    logger.info(f"    Embedding {len(docs)} documents...")

    batch_size = DEFAULT_BATCH_SIZE
    try:
        vectordb = get_vector_store()
    except (ImportError, ValueError) as exc:
        logger.error(str(exc))
        return
    if backend == "pgvector":
        vectordb.delete()

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        logger.debug(
            f"    Batch {i // batch_size + 1}/{(len(docs) - 1) // batch_size + 1} ({len(batch)} docs)"
        )

        max_retries = 5
        for attempt in range(max_retries):
            try:
                ids = [str(doc.metadata["doi"]) for doc in batch]
                vectordb.add_documents(batch, ids=ids)
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    wait_time = (2**attempt) + 1
                    logger.warning(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise
        else:
            logger.error(f"Failed after {max_retries} retries, skipping batch")

    if backend == "chroma":
        logger.info(f"    Done! Vector store saved to {chroma_dir}/")
    else:
        logger.info("    Done! Vector store saved to Postgres (pgvector)")
