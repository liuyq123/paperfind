"""Vector database functions for Zotero sync."""

from typing import Any, Dict, List, Set

from langchain_core.documents import Document

from paperfind.logging import get_logger
from paperfind.vectorstore import get_vector_store, get_vector_store_backend

logger = get_logger(__name__)


def get_vectordb():
    """Get or create vector store."""
    return get_vector_store("zotero")


def build_document_for_item(item: Dict[str, Any]) -> Document:
    """Build a LangChain document from an item dict."""
    zotero_key = item["zotero_key"]
    title = item.get("title") or ""
    authors = item.get("authors") or ""
    abstract = item.get("abstract") or ""
    tags = item.get("tags", [])

    parts = [title]
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if tags:
        parts.append("Tags: " + ", ".join(tags))
    page_content = "\n\n".join(parts).strip()

    metadata = {
        "zotero_key": zotero_key,
        "title": title,
        "authors": authors,
        "abstract": abstract,
    }

    return Document(page_content=page_content, metadata=metadata)


def get_embedded_zotero_keys() -> Set[str]:
    """Get set of zotero_keys that are already embedded in the vector store."""
    try:
        vectordb = get_vectordb()
    except (ImportError, ValueError) as exc:
        logger.error(str(exc))
        return set()

    backend = get_vector_store_backend()
    if backend == "pgvector" and hasattr(vectordb, "list_ids"):
        try:
            return set(vectordb.list_ids())
        except Exception as exc:
            logger.error(f"Error retrieving pgvector ids: {exc}")
            return set()

    # Get all documents from the vector store
    # For Chroma, we can use get() to retrieve all
    try:
        # Try to get all IDs (which are zotero_keys now)
        result = vectordb.get()
        if result and "ids" in result:
            return set(result["ids"])
    except Exception:
        pass

    return set()


def is_item_embedded(zotero_key: str) -> bool:
    """Check if an item is already embedded in the vector store."""
    try:
        vectordb = get_vectordb()
        backend = get_vector_store_backend()
        if backend == "pgvector" and hasattr(vectordb, "has_id"):
            return bool(vectordb.has_id(zotero_key))
        result = vectordb.get(ids=[zotero_key])
        return bool(result and result.get("ids"))
    except Exception:
        return False


def embed_items(items: List[Dict[str, Any]], skip_existing: bool = True) -> int:
    """
    Embed items to the vector store.

    Args:
        items: List of item dicts from the database
        skip_existing: If True, skip items that are already embedded

    Returns:
        Number of items embedded.
    """
    try:
        vectordb = get_vectordb()
    except (ImportError, ValueError) as exc:
        logger.error(str(exc))
        return 0

    # Get already embedded keys if skipping existing
    existing_keys: Set[str] = set()
    if skip_existing:
        existing_keys = get_embedded_zotero_keys()
        if existing_keys:
            logger.info(f"Found {len(existing_keys)} items already embedded")

    # Filter and build documents
    docs_to_embed: List[Document] = []
    ids_to_embed: List[str] = []

    for item in items:
        zotero_key = item["zotero_key"]

        # Skip if already embedded
        if skip_existing and zotero_key in existing_keys:
            continue

        # Skip items without content
        if not item.get("title") and not item.get("abstract"):
            continue

        doc = build_document_for_item(item)
        docs_to_embed.append(doc)
        ids_to_embed.append(zotero_key)

    if not docs_to_embed:
        logger.info("No new items to embed")
        return 0

    # Embed in batches
    batch_size = 100
    total_embedded = 0

    for i in range(0, len(docs_to_embed), batch_size):
        batch_docs = docs_to_embed[i : i + batch_size]
        batch_ids = ids_to_embed[i : i + batch_size]

        try:
            vectordb.add_documents(batch_docs, ids=batch_ids)
            total_embedded += len(batch_docs)
            logger.info(f"Embedded {total_embedded}/{len(docs_to_embed)} items")
        except Exception as exc:
            logger.error(f"Error embedding batch: {exc}")
            continue

    return total_embedded


def delete_item_embeddings(zotero_keys: List[str]) -> int:
    """
    Delete embeddings for specific items.

    Args:
        zotero_keys: List of zotero_keys to delete

    Returns:
        Number of embeddings deleted.
    """
    try:
        vectordb = get_vectordb()
    except (ImportError, ValueError) as exc:
        logger.error(str(exc))
        return 0

    try:
        vectordb.delete(ids=zotero_keys)
        return len(zotero_keys)
    except Exception as exc:
        logger.error(f"Error deleting embeddings: {exc}")
        return 0
