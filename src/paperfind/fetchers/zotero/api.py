"""Zotero API functions."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from paperfind.logging import get_logger

logger = get_logger(__name__)

# Type aliases for Zotero API responses
ZoteroCollection = Dict[str, Any]
ZoteroItem = Dict[str, Any]
ZoteroItemRow = Dict[str, Any]


def _zotero_base(library_id: str, library_type: str) -> str:
    """Get base URL for Zotero API."""
    prefix = "users" if library_type == "user" else "groups"
    return f"https://api.zotero.org/{prefix}/{library_id}"


def fetch_collections(
    library_id: str,
    api_key: str,
    library_type: str = "user",
    limit: int = 100,
) -> List[ZoteroCollection]:
    """Fetch all collections from Zotero library."""
    base = _zotero_base(library_id, library_type) + "/collections"
    headers = {"Zotero-API-Key": api_key}
    collections: List[ZoteroCollection] = []
    start = 0

    while True:
        params = {
            "format": "json",
            "limit": limit,
            "start": start,
            "sort": "title",
            "direction": "asc",
        }
        r = requests.get(base, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        collections.extend(batch)
        if len(batch) < limit:
            break
        start += limit

    return collections


def resolve_collection_name_to_key(
    library_id: str,
    api_key: str,
    library_type: str,
    collection_name: str,
) -> Optional[str]:
    """Resolve collection name to Zotero collection key."""
    cols = fetch_collections(library_id, api_key, library_type)
    matches = [
        c for c in cols
        if c.get("data", {}).get("name", "").strip().lower()
        == collection_name.strip().lower()
    ]

    if len(matches) == 1:
        return matches[0]["data"]["key"]
    elif len(matches) == 0:
        logger.warning(f"No collection named '{collection_name}' found")
        return None
    else:
        logger.warning(f"Multiple collections named '{collection_name}' found:")
        for c in matches:
            logger.warning(f"  - key: {c['data']['key']}")
        return None


def fetch_items_for_project(
    library_id: str,
    api_key: str,
    library_type: str = "user",
    collection_key: Optional[str] = None,
    limit: int = 100,
) -> List[ZoteroItem]:
    """Fetch all items from Zotero library or collection."""
    root = _zotero_base(library_id, library_type)
    headers = {"Zotero-API-Key": api_key}
    items: List[ZoteroItem] = []
    start = 0

    while True:
        params = {
            "format": "json",
            "limit": limit,
            "start": start,
            "sort": "dateModified",
            "direction": "asc",
        }

        if collection_key:
            url = f"{root}/collections/{collection_key}/items"
        else:
            url = f"{root}/items"

        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        items.extend(batch)
        if len(batch) < limit:
            break
        start += limit

    return items


def fetch_library_items(
    library_id: str,
    api_key: str,
    library_type: str = "user",
    limit: int = 100,
) -> List[ZoteroItem]:
    """Fetch all items from Zotero library (entire library, not a specific collection)."""
    root = _zotero_base(library_id, library_type)
    headers = {"Zotero-API-Key": api_key}
    items: List[ZoteroItem] = []
    start = 0

    while True:
        params = {
            "format": "json",
            "limit": limit,
            "start": start,
            "sort": "dateModified",
            "direction": "asc",
        }
        url = f"{root}/items"

        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        items.extend(batch)
        if len(batch) < limit:
            break
        start += limit

    return items


def zotero_item_to_row(item: ZoteroItem) -> ZoteroItemRow:
    """Convert Zotero API item to database row."""
    data = item.get("data", {})

    title = data.get("title") or ""
    creators = data.get("creators") or []
    authors = []
    for c in creators:
        first = c.get("firstName", "") or ""
        last = c.get("lastName", "") or ""
        name = f"{first} {last}".strip()
        if name:
            authors.append(name)
    authors_str = ", ".join(authors)

    abstract = data.get("abstractNote") or ""
    doi = data.get("DOI") or None
    date = data.get("date") or None
    url = data.get("url") or None
    item_type = data.get("itemType") or None
    zotero_key = data.get("key")
    tags = [t.get("tag") for t in data.get("tags", []) if t.get("tag")]
    # Collection keys this item belongs to
    collections = data.get("collections", [])

    return {
        "zotero_key": zotero_key,
        "title": title,
        "authors": authors_str,
        "abstract": abstract,
        "doi": doi,
        "date": date,
        "url": url,
        "item_type": item_type,
        "tags": tags,
        "collections": collections,
    }
