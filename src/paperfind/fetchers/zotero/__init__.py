"""Zotero sync module.

Provides functions to sync Zotero library to local SQLite and vector DB.
"""

from .api import fetch_collections, fetch_items_for_project
from .db import get_conn, get_or_create_project, init_db
from .sync import list_collections, run_sync, rebuild_all_vectors, sync_project
from .vector import get_vectordb, rebuild_vectors_for_project

__all__ = [
    # Database
    "init_db",
    "get_conn",
    "get_or_create_project",
    # API
    "fetch_collections",
    "fetch_items_for_project",
    # Vector
    "get_vectordb",
    "rebuild_vectors_for_project",
    # Sync
    "sync_project",
    "list_collections",
    "rebuild_all_vectors",
    "run_sync",
]
