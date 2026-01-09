"""Shared utilities for search-related modules."""


from typing import Any

from paperfind.db import is_postgres
from paperfind.logging import get_logger
from paperfind.vectorstore import get_vector_store_backend, vector_store_exists

logger = get_logger(__name__)


def check_vector_store(source: str = "daily_papers") -> bool:
    """Check if the vector store exists for the given source."""
    backend = get_vector_store_backend()
    if backend == "pgvector" and not is_postgres():
        logger.error(
            "pgvector backend requires PAPERFIND_DB_URL to be set. "
            "Example: PAPERFIND_DB_URL=postgresql://user:pass@localhost/paperfind"
        )
        return False

    if not vector_store_exists(source):
        if source == "zotero":
            logger.error("No Zotero embeddings found. Run 'paperfind sync && paperfind embed' first.")
        else:
            logger.error("No paper embeddings found. Run 'paperfind fetch --rebuild-vectors' first.")
        return False
    return True


def warn_if_empty(vectordb: Any, source: str = "daily_papers") -> None:
    """Warn if the vector store exists but has no documents."""
    count = None
    if hasattr(vectordb, "count"):
        try:
            count = vectordb.count()
        except Exception:
            count = None

    if count is None:
        collection = getattr(vectordb, "_collection", None)
        if collection is None:
            return
        try:
            count = collection.count()
        except Exception:
            return

    if count == 0:
        if source == "zotero":
            logger.warning("Zotero vector store is empty. Run 'paperfind sync && paperfind embed' first.")
        else:
            logger.warning(
                "Paper embeddings store is empty. "
                "Run 'paperfind fetch --rebuild-vectors' first."
            )
