"""Vector database functions for Zotero sync."""

from typing import List

from langchain_core.documents import Document

from paperfind.db import ZOTERO_SCHEMA, placeholder, qualify_table
from paperfind.logging import get_logger
from paperfind.vectorstore import get_vector_store

from .db import get_conn

logger = get_logger(__name__)


def get_vectordb():
    """Get or create vector store."""
    return get_vector_store("zotero")


def build_docs_for_project(project_id: int) -> List[Document]:
    """Build LangChain documents from project items."""
    conn = get_conn()
    cur = conn.cursor()
    items_table = qualify_table(ZOTERO_SCHEMA, "items")
    tags_table = qualify_table(ZOTERO_SCHEMA, "tags")
    ph = placeholder()
    cur.execute(
        f"""
        SELECT id, zotero_key, title, abstract
        FROM {items_table}
        WHERE project_id = {ph}
        """,
        (project_id,),
    )
    rows = cur.fetchall()

    docs: List[Document] = []

    for row in rows:
        item_id = row["id"]
        zotero_key = row["zotero_key"]
        title = row["title"]
        abstract = row["abstract"]
        # Fetch tags
        cur.execute(f"SELECT tag FROM {tags_table} WHERE item_id = {ph}", (item_id,))
        tags = [r["tag"] for r in cur.fetchall()]

        parts = [title or ""]
        if abstract:
            parts.append(f"Abstract: {abstract}")
        if tags:
            parts.append("Tags: " + ", ".join(tags))
        page_content = "\n\n".join(parts).strip()
        if not page_content:
            continue

        metadata = {
            "project_id": project_id,
            "item_id": item_id,
            "zotero_key": zotero_key,
        }
        docs.append(Document(page_content=page_content, metadata=metadata))

    conn.close()
    return docs


def rebuild_vectors_for_project(project_id: int) -> int:
    """Rebuild vector embeddings for a project."""
    try:
        vectordb = get_vectordb()
    except (ImportError, ValueError) as exc:
        logger.error(str(exc))
        return 0

    # Delete existing vectors for this project
    vectordb.delete(where={"project_id": project_id})

    docs = build_docs_for_project(project_id)
    if docs:
        ids = [str(doc.metadata["item_id"]) for doc in docs]
        vectordb.add_documents(docs, ids=ids)

    return len(docs)
