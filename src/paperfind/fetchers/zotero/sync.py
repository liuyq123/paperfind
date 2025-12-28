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

from .api import (
    fetch_collections,
    fetch_items_for_project,
    resolve_collection_name_to_key,
)
from .db import get_conn, get_or_create_project, init_db, replace_project_items
from .vector import rebuild_vectors_for_project


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

    print(f"Fetched {len(items)} items from Zotero")
    item_ids = replace_project_items(project_id, items)
    print(f"Stored {len(item_ids)} items in database (project_id={project_id})")

    num_docs = rebuild_vectors_for_project(project_id)
    print(f"Built {num_docs} vector embeddings")

    return project_id


def list_collections() -> None:
    """List all collections in the Zotero library."""
    if not ZOTERO_API_KEY or not ZOTERO_USER_ID:
        print("Error: ZOTERO_API_KEY and ZOTERO_USER_ID must be set in .env")
        return

    collections = fetch_collections(
        library_id=ZOTERO_USER_ID,
        api_key=ZOTERO_API_KEY,
        library_type=ZOTERO_LIBRARY_TYPE,
    )

    if not collections:
        print("No collections found in your Zotero library.")
        return

    print(f"\nFound {len(collections)} collections:\n")
    for c in collections:
        name = c.get("data", {}).get("name", "Unknown")
        num_items = c.get("meta", {}).get("numItems", 0)
        print(f"  - {name} ({num_items} items)")


def rebuild_all_vectors() -> None:
    """Rebuild vectors for all projects."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM projects")
    projects = cur.fetchall()
    conn.close()

    if not projects:
        print("No projects found. Run sync first.")
        return

    for project_id, name in projects:
        print(f"\nRebuilding vectors for '{name}' (project_id={project_id})...")
        num_docs = rebuild_vectors_for_project(project_id)
        print(f"  Built {num_docs} embeddings")


def run_sync(collection: Optional[str], list_collections_flag: bool) -> None:
    """Run a Zotero sync with parsed parameters."""
    # Initialize database
    init_db()

    if list_collections_flag:
        list_collections()
        return

    # Project name = collection name (or "whole library")
    project_name = collection if collection else "whole library"

    print(f"\n{'='*60}")
    print("Syncing Zotero Library")
    print(f"{'='*60}")

    if collection:
        print(f"Collection: {collection}")
    else:
        print("Syncing entire library")

    print()

    sync_project(
        project_name=project_name,
        collection_name=collection,
    )

    print("\nSync complete!")
