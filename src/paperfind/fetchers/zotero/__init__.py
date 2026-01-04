"""Zotero sync module.

Provides functions to sync Zotero library to local SQLite and vector DB.
"""

from .api import fetch_collections, fetch_library_items
from .db import (
    get_conn,
    get_or_create_library,
    init_db,
    get_all_collections,
    get_items_for_collection,
    get_collection_by_name_or_key,
)
from .sync import (
    embed_collection,
    list_collections,
    run_embed,
    run_sync,
    sync_library,
)
from .vector import embed_items, get_vectordb

__all__ = [
    # Database
    "init_db",
    "get_conn",
    "get_or_create_library",
    "get_all_collections",
    "get_items_for_collection",
    "get_collection_by_name_or_key",
    # API
    "fetch_collections",
    "fetch_library_items",
    # Vector
    "get_vectordb",
    "embed_items",
    # Sync
    "sync_library",
    "embed_collection",
    "list_collections",
    "run_sync",
    "run_embed",
]
