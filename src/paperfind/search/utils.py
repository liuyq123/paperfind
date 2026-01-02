"""Shared utilities for search-related modules."""

from pathlib import Path

from paperfind.config import get_chroma_store_dir, get_zotero_vectors_dir


def check_vector_store(source: str = "daily_papers") -> bool:
    """Check if the vector store exists for the given source."""
    if source == "zotero":
        store_dir = Path(get_zotero_vectors_dir())
        if not store_dir.exists():
            print("Error: No Zotero embeddings found. Run 'paperfind sync' first.")
            return False
    else:
        store_dir = Path(get_chroma_store_dir())
        if not store_dir.exists():
            print("Error: No paper embeddings found. Run 'paperfind fetch --rebuild-vectors' first.")
            return False
    return True


def warn_if_empty(vectordb, source: str = "daily_papers") -> None:
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
            print("Warning: Zotero vector store is empty. Run 'paperfind sync' first.")
        else:
            print(
                "Warning: Paper embeddings store is empty. "
                "Run 'paperfind fetch --rebuild-vectors' first."
            )
