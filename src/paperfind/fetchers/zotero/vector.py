"""Vector database functions for Zotero sync."""

from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from paperfind.config import EMBEDDING_MODEL, ZOTERO_VECTORS_DIR

from .db import get_conn

# ChromaDB collection name
CHROMA_COLLECTION = "zotero_all"


def get_vectordb() -> Chroma:
    """Get or create ChromaDB vector store."""
    Path(ZOTERO_VECTORS_DIR).mkdir(parents=True, exist_ok=True)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=ZOTERO_VECTORS_DIR,
        collection_name=CHROMA_COLLECTION,
    )
    return vectordb


def build_docs_for_project(project_id: int) -> List[Document]:
    """Build LangChain documents from project items."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, zotero_key, title, abstract
        FROM items
        WHERE project_id = ?
        """,
        (project_id,),
    )
    rows = cur.fetchall()

    docs: List[Document] = []

    for item_id, zotero_key, title, abstract in rows:
        # Fetch tags
        cur.execute("SELECT tag FROM tags WHERE item_id = ?", (item_id,))
        tags = [r[0] for r in cur.fetchall()]

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
    vectordb = get_vectordb()

    # Delete existing vectors for this project
    vectordb.delete(where={"project_id": project_id})

    docs = build_docs_for_project(project_id)
    if docs:
        vectordb.add_documents(docs)

    return len(docs)
