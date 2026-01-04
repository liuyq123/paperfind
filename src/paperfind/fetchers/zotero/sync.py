"""
High-level Zotero sync functions and CLI.

Usage:
    paperfind sync                           # Sync entire library
    paperfind sync --list-collections        # List available collections
    paperfind embed                          # Embed all items in library
    paperfind embed "my collection"          # Embed items in a collection
"""

from typing import Optional

from paperfind.config import (
    ZOTERO_API_KEY,
    ZOTERO_LIBRARY_TYPE,
    ZOTERO_USER_ID,
)
from paperfind.logging import get_logger

from .api import fetch_collections, fetch_library_items
from .db import (
    get_all_collections,
    get_all_items,
    get_collection_by_name_or_key,
    get_items_for_collection,
    get_library,
    get_or_create_library,
    init_db,
    update_library_sync_time,
    upsert_collections,
    upsert_item,
)
from .vector import embed_items

logger = get_logger(__name__)


def sync_library() -> int:
    """
    Sync entire Zotero library to local database.

    Returns the number of items synced.
    """
    if not ZOTERO_API_KEY:
        raise ValueError("ZOTERO_API_KEY is not set in .env")
    if not ZOTERO_USER_ID:
        raise ValueError("ZOTERO_USER_ID is not set in .env")

    # Get or create library record
    library_pk = get_or_create_library(
        library_id=ZOTERO_USER_ID,
        library_type=ZOTERO_LIBRARY_TYPE,
    )

    # Fetch all collections from Zotero
    logger.info("Fetching collections from Zotero...")
    collections = fetch_collections(
        library_id=ZOTERO_USER_ID,
        api_key=ZOTERO_API_KEY,
        library_type=ZOTERO_LIBRARY_TYPE,
    )
    logger.info(f"Found {len(collections)} collections")

    # Upsert collections to DB
    collection_key_to_id = upsert_collections(library_pk, collections)

    # Fetch all items from Zotero
    logger.info("Fetching items from Zotero...")
    items = fetch_library_items(
        library_id=ZOTERO_USER_ID,
        api_key=ZOTERO_API_KEY,
        library_type=ZOTERO_LIBRARY_TYPE,
    )
    logger.info(f"Fetched {len(items)} items from Zotero")

    # Upsert each item
    item_count = 0
    for item in items:
        item_id = upsert_item(library_pk, item, collection_key_to_id)
        if item_id is not None:
            item_count += 1

    logger.info(f"Synced {item_count} items to database")

    # Update sync time
    update_library_sync_time(library_pk)

    return item_count


def embed_all(force: bool = False) -> int:
    """
    Embed all items in the Zotero library.

    Args:
        force: If True, re-embed all items even if already embedded

    Returns:
        Number of items embedded.
    """
    if not ZOTERO_USER_ID:
        raise ValueError("ZOTERO_USER_ID is not set in .env")

    # Get library
    library = get_library(ZOTERO_USER_ID, ZOTERO_LIBRARY_TYPE)
    if not library:
        raise ValueError("Library not found. Run 'paperfind sync' first.")

    library_pk = library["id"]

    # Get all items
    items = get_all_items(library_pk)
    if not items:
        logger.warning("No items found in library")
        return 0

    logger.info(f"Found {len(items)} items in library")

    # Embed items
    num_embedded = embed_items(items, skip_existing=not force)

    return num_embedded


def embed_collection(
    collection_name_or_key: str,
    force: bool = False,
) -> int:
    """
    Embed items in a specific collection.

    Args:
        collection_name_or_key: Collection name or key to embed
        force: If True, re-embed all items even if already embedded

    Returns:
        Number of items embedded.
    """
    if not ZOTERO_USER_ID:
        raise ValueError("ZOTERO_USER_ID is not set in .env")

    # Get library
    library = get_library(ZOTERO_USER_ID, ZOTERO_LIBRARY_TYPE)
    if not library:
        raise ValueError("Library not found. Run 'paperfind sync' first.")

    library_pk = library["id"]

    # Get collection
    collection = get_collection_by_name_or_key(library_pk, collection_name_or_key)
    if not collection:
        raise ValueError(f"Collection '{collection_name_or_key}' not found.")

    collection_id = collection["id"]
    collection_name = collection["name"]

    # Get items in collection
    items = get_items_for_collection(collection_id)
    if not items:
        logger.warning(f"No items found in collection '{collection_name}'")
        return 0

    logger.info(f"Found {len(items)} items in collection '{collection_name}'")

    # Embed items
    num_embedded = embed_items(items, skip_existing=not force)

    return num_embedded


def list_collections() -> None:
    """List all collections in the Zotero library."""
    if not ZOTERO_USER_ID:
        logger.error("ZOTERO_USER_ID must be set in .env")
        return

    # Try to get from local DB first
    library = get_library(ZOTERO_USER_ID, ZOTERO_LIBRARY_TYPE)

    if library:
        collections = get_all_collections(library["id"])
        if collections:
            logger.info(f"Found {len(collections)} collections:")
            for c in collections:
                name = c.get("name", "Unknown")
                num_items = c.get("num_items", 0)
                logger.info(f"  - {name} ({num_items} items)")
            return

    # Fall back to API if no local data
    if not ZOTERO_API_KEY:
        logger.error("ZOTERO_API_KEY must be set in .env")
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


def run_sync(list_collections_flag: bool) -> None:
    """Run a Zotero sync with parsed parameters."""
    import sys

    # Initialize database
    init_db()

    if list_collections_flag:
        list_collections()
        return

    logger.info("=" * 60)
    logger.info("Syncing Zotero Library")
    logger.info("=" * 60)

    try:
        item_count = sync_library()
        logger.info(f"Sync complete! Synced {item_count} items.")
    except ValueError as e:
        logger.error(f"Sync failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during sync: {e}")
        sys.exit(1)


def run_embed(collection: Optional[str], force: bool) -> None:
    """Run embedding for a collection or entire library."""
    import sys

    # Initialize database (in case it hasn't been)
    init_db()

    if collection:
        logger.info("=" * 60)
        logger.info(f"Embedding Collection: {collection}")
        logger.info("=" * 60)

        try:
            num_embedded = embed_collection(collection, force=force)
            logger.info(f"Embedding complete! Embedded {num_embedded} items.")
        except ValueError as e:
            logger.error(f"Embedding failed: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error during embedding: {e}")
            sys.exit(1)
    else:
        logger.info("=" * 60)
        logger.info("Embedding All Items")
        logger.info("=" * 60)

        try:
            num_embedded = embed_all(force=force)
            logger.info(f"Embedding complete! Embedded {num_embedded} items.")
        except ValueError as e:
            logger.error(f"Embedding failed: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Unexpected error during embedding: {e}")
            sys.exit(1)
