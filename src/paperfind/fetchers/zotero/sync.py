"""
High-level Zotero sync functions and CLI.

Usage:
    paperfind sync                           # Sync entire library
    paperfind sync --collection "my papers"  # Sync specific collection
    paperfind sync --list-collections        # List available collections
"""

from typing import Optional

from paperfind.config import (
    ZOTERO_API_KEY,
    ZOTERO_LIBRARY_TYPE,
    ZOTERO_USER_ID,
)
from paperfind.logging import get_logger

from .api import (
    fetch_collections,
    fetch_items_for_project,
    resolve_collection_name_to_key,
)
from paperfind.db import ZOTERO_SCHEMA, qualify_table

from .db import get_conn, get_or_create_project, init_db, replace_project_items
from .vector import rebuild_vectors_for_project

logger = get_logger(__name__)


def sync_project(
    project_name: str,
    collection_name: Optional[str] = None,
) -> int:
    """
    Sync a Zotero library or collection to the local database.

    Returns the project_id.
    """
    if not ZOTERO_API_KEY:
        raise ValueError("ZOTERO_API_KEY is not set in .env")
    if not ZOTERO_USER_ID:
        raise ValueError("ZOTERO_USER_ID is not set in .env")

    collection_key = None
    if collection_name:
        collection_key = resolve_collection_name_to_key(
            library_id=ZOTERO_USER_ID,
            api_key=ZOTERO_API_KEY,
            library_type=ZOTERO_LIBRARY_TYPE,
            collection_name=collection_name,
        )
        if collection_key is None:
            raise ValueError(f"Could not resolve collection name '{collection_name}'")

    project_id = get_or_create_project(
        name=project_name,
        library_id=ZOTERO_USER_ID,
        library_type=ZOTERO_LIBRARY_TYPE,
        collection_name=collection_name,
        collection_key=collection_key,
    )

    items = fetch_items_for_project(
        library_id=ZOTERO_USER_ID,
        api_key=ZOTERO_API_KEY,
        library_type=ZOTERO_LIBRARY_TYPE,
        collection_key=collection_key,
    )

    logger.info(f"Fetched {len(items)} items from Zotero")
    item_ids = replace_project_items(project_id, items)
    logger.info(f"Stored {len(item_ids)} items in database (project_id={project_id})")

    num_docs = rebuild_vectors_for_project(project_id)
    logger.info(f"Built {num_docs} vector embeddings")

    return project_id


def list_collections() -> None:
    """List all collections in the Zotero library."""
    if not ZOTERO_API_KEY or not ZOTERO_USER_ID:
        logger.error("ZOTERO_API_KEY and ZOTERO_USER_ID must be set in .env")
        return

    collections = fetch_collections(
        library_id=ZOTERO_USER_ID,
        api_key=ZOTERO_API_KEY,
        library_type=ZOTERO_LIBRARY_TYPE,
    )

    if not collections:
        logger.info("No collections found in your Zotero library.")
        return

    logger.info(f"Found {len(collections)} collections:")
    for c in collections:
        name = c.get("data", {}).get("name", "Unknown")
        num_items = c.get("meta", {}).get("numItems", 0)
        logger.info(f"  - {name} ({num_items} items)")


def rebuild_all_vectors() -> None:
    """Rebuild vectors for all projects."""
    conn = get_conn()
    cur = conn.cursor()
    table = qualify_table(ZOTERO_SCHEMA, "projects")
    cur.execute(f"SELECT id, name FROM {table}")
    projects = cur.fetchall()
    conn.close()

    if not projects:
        logger.warning("No projects found. Run sync first.")
        return

    for row in projects:
        project_id = row["id"]
        name = row["name"]
        logger.info(f"Rebuilding vectors for '{name}' (project_id={project_id})...")
        num_docs = rebuild_vectors_for_project(project_id)
        logger.info(f"  Built {num_docs} embeddings")


def run_sync(collection: Optional[str], list_collections_flag: bool) -> None:
    """Run a Zotero sync with parsed parameters."""
    # Initialize database
    init_db()

    if list_collections_flag:
        list_collections()
        return

    # Project name = collection name (or "whole library")
    project_name = collection if collection else "whole library"

    logger.info("=" * 60)
    logger.info("Syncing Zotero Library")
    logger.info("=" * 60)

    if collection:
        logger.info(f"Collection: {collection}")
    else:
        logger.info("Syncing entire library")

    sync_project(
        project_name=project_name,
        collection_name=collection,
    )

    logger.info("Sync complete!")
