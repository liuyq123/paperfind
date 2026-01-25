"""Vector store helpers for paper fetchers."""


import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from paperfind.config import DAILY_PAPERS_DB, get_chroma_store_dir
from paperfind.db import (
    DAILY_SCHEMA,
    get_db,
    is_postgres,
    placeholders,
    qualify_table,
    table_exists,
)
from paperfind.logging import get_logger
from paperfind.vectorstore import get_existing_ids, get_vector_store, get_vector_store_backend

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 100


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""

    pass


def _build_documents(rows: Sequence[Mapping[str, Any]]) -> list[Any]:
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
            "title": title,
            "authors": row["authors"],
            "abstract": abstract or None,
            "created_date": row["created_date"],
            "type": row["type"],
            "source": row["source"],
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def upsert_vectors_for_dois(
    dois: list[str], batch_size: int = DEFAULT_BATCH_SIZE, raise_on_error: bool = False
) -> int:
    """Upsert vector embeddings for specific DOIs.

    Skips DOIs that are already in the vector store to avoid re-embedding.

    Args:
        dois: List of DOIs to embed.
        batch_size: Number of documents to process per batch.
        raise_on_error: If True, raise VectorStoreError on initialization failures.
                        If False (default), log error and return 0.

    Returns:
        Number of documents embedded.

    Raises:
        VectorStoreError: If raise_on_error is True and vector store init fails.
    """
    if not dois:
        return 0

    logger.info("[Vectors] Updating vector embeddings for new papers...")

    try:
        vectordb = get_vector_store()
    except (ImportError, ValueError) as exc:
        logger.error(str(exc))
        if raise_on_error:
            raise VectorStoreError(str(exc)) from exc
        return 0

    # Filter out DOIs already in the vector store
    existing_ids = get_existing_ids(vectordb)
    new_dois = [doi for doi in dois if doi not in existing_ids]

    if not new_dois:
        logger.info("    No new papers to embed (all already in vector store)")
        return 0

    skipped = len(dois) - len(new_dois)
    if skipped > 0:
        logger.debug(f"    Skipping {skipped} papers already in vector store")

    total_docs = 0

    with get_db(DAILY_SCHEMA) as conn:
        cur = conn.cursor()

        for i in range(0, len(new_dois), batch_size):
            chunk = new_dois[i : i + batch_size]
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

            max_retries = 5
            for attempt in range(max_retries):
                try:
                    vectordb.add_documents(docs, ids=ids)
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
                continue

            total_docs += len(docs)

    logger.info(f"    Embedded {total_docs} new documents into vector store")
    return total_docs


def rebuild_vectors(raise_on_error: bool = False) -> bool:
    """Rebuild the vector database from the papers database.

    Args:
        raise_on_error: If True, raise exceptions on failures.
                        If False (default), log errors and return False.

    Returns:
        True if rebuild succeeded, False otherwise.

    Raises:
        VectorStoreError: If raise_on_error is True and an error occurs.
    """
    import shutil

    logger.info("[Vectors] Rebuilding vector embeddings...")

    # Check if database exists before proceeding (SQLite only)
    if not is_postgres() and not Path(DAILY_PAPERS_DB).exists():
        msg = "No papers database found. Run 'paperfind fetch' first."
        logger.error(msg)
        if raise_on_error:
            raise VectorStoreError(msg)
        return False

    backend = get_vector_store_backend()
    chroma_dir = get_chroma_store_dir() if backend == "chroma" else None

    try:
        with get_db(DAILY_SCHEMA) as conn:
            if not table_exists(conn, DAILY_SCHEMA, "works"):
                msg = "No papers in database. Run 'paperfind fetch' first."
                logger.error(msg)
                if raise_on_error:
                    raise VectorStoreError(msg)
                return False

            if backend == "chroma" and chroma_dir and Path(chroma_dir).exists():
                shutil.rmtree(chroma_dir)
                logger.debug(f"Cleared existing store at {chroma_dir}/")

            table = qualify_table(DAILY_SCHEMA, "works")
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) AS cnt FROM {table}")
            total = cur.fetchone()["cnt"]
            if not total:
                logger.warning("No documents to embed")
                return True  # Not an error, just nothing to do

            logger.info(f"    Embedding {total} documents...")

            batch_size = DEFAULT_BATCH_SIZE
            try:
                vectordb = get_vector_store()
            except (ImportError, ValueError) as exc:
                logger.error(str(exc))
                if raise_on_error:
                    raise VectorStoreError(str(exc)) from exc
                return False

            if backend == "pgvector":
                vectordb.delete()

            cur.execute(
                f"SELECT doi, title, authors, abstract, created_date, type, source FROM {table}"
            )
            batch_index = 0
            while True:
                rows = cur.fetchmany(batch_size)
                if not rows:
                    break

                batch = _build_documents(rows)
                if not batch:
                    logger.warning("No documents to embed")
                    break

                batch_index += 1
                logger.debug(f"    Batch {batch_index} ({len(batch)} docs)")

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

    except Exception as exc:
        if raise_on_error:
            raise
        logger.error(f"Failed to connect to database: {exc}")
        return False

    if backend == "chroma" and chroma_dir:
        logger.info(f"    Done! Vector store saved to {chroma_dir}/")
    else:
        logger.info("    Done! Vector store saved to Postgres (pgvector)")

    return True


def prune_vectors(
    dois: list[str], batch_size: int = DEFAULT_BATCH_SIZE, raise_on_error: bool = False
) -> int:
    """Delete vector embeddings for specific DOIs.

    Args:
        dois: List of DOIs to delete from the vector store.
        batch_size: Number of IDs to delete per batch.
        raise_on_error: If True, raise VectorStoreError on initialization failures.
                        If False (default), log error and return 0.

    Returns:
        Number of embeddings deleted.

    Raises:
        VectorStoreError: If raise_on_error is True and vector store init fails.
    """
    if not dois:
        return 0

    try:
        vectordb = get_vector_store()
    except (ImportError, ValueError) as exc:
        logger.error(str(exc))
        if raise_on_error:
            raise VectorStoreError(str(exc)) from exc
        return 0

    # Get existing IDs to only delete those that exist
    existing_ids = get_existing_ids(vectordb)
    dois_to_delete = [doi for doi in dois if doi in existing_ids]

    if not dois_to_delete:
        logger.info("No vector embeddings to delete")
        return 0

    # Delete in batches
    total_deleted = 0
    for i in range(0, len(dois_to_delete), batch_size):
        batch = dois_to_delete[i : i + batch_size]
        try:
            vectordb.delete(ids=batch)
            total_deleted += len(batch)
        except Exception as exc:
            logger.error(f"Error deleting vector batch: {exc}")
            continue

    logger.info(f"Deleted {total_deleted} embeddings from vector store")
    return total_deleted
