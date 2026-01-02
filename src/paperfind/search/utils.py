"""Shared utilities for search-related modules."""

from pathlib import Path
from typing import Any

from paperfind.config import get_chroma_store_dir, get_zotero_vectors_dir
from paperfind.logging import get_logger

logger = get_logger(__name__)


def check_vector_store(source: str = "daily_papers") -> bool:
    """Check if the vector store exists for the given source."""
    if source == "zotero":
        store_dir = Path(get_zotero_vectors_dir())
        if not store_dir.exists():
            logger.error("No Zotero embeddings found. Run 'paperfind sync' first.")
            return False
    else:
        store_dir = Path(get_chroma_store_dir())
        if not store_dir.exists():
            logger.error("No paper embeddings found. Run 'paperfind fetch --rebuild-vectors' first.")
            return False
    return True


def warn_if_empty(vectordb: Any, source: str = "daily_papers") -> None:
    """Warn if the vector store exists but has no documents."""
    collection = getattr(vectordb, "_collection", None)
    if collection is None:
        return

    try:
        count = collection.count()
    except Exception:
        return

    if count == 0:
        if source == "zotero":
            logger.warning("Zotero vector store is empty. Run 'paperfind sync' first.")
        else:
            logger.warning(
                "Paper embeddings store is empty. "
                "Run 'paperfind fetch --rebuild-vectors' first."
            )
