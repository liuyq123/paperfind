"""Database functions for Zotero sync."""

from typing import Any, Dict, List, Optional

from paperfind.db import (
    ZOTERO_SCHEMA,
    get_conn as get_db_conn,
    is_postgres,
    placeholder,
    placeholders,
    qualify_table,
)

# Type alias for Zotero items from API
ZoteroItem = Dict[str, Any]


def get_conn():
    """Get database connection."""
    return get_db_conn(ZOTERO_SCHEMA)


def init_db() -> None:
    """Initialize the database schema."""
    conn = get_conn()
    cur = conn.cursor()

    libraries_table = qualify_table(ZOTERO_SCHEMA, "libraries")
    collections_table = qualify_table(ZOTERO_SCHEMA, "collections")
    items_table = qualify_table(ZOTERO_SCHEMA, "items")
    item_collections_table = qualify_table(ZOTERO_SCHEMA, "item_collections")
    tags_table = qualify_table(ZOTERO_SCHEMA, "tags")

    if is_postgres():
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {ZOTERO_SCHEMA}")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {libraries_table} (
                id                SERIAL PRIMARY KEY,
                library_id        TEXT NOT NULL,
                library_type      TEXT NOT NULL,
                last_synced_at    TIMESTAMP,
                UNIQUE(library_id, library_type)
            );
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {collections_table} (
                id                SERIAL PRIMARY KEY,
                library_pk        INTEGER NOT NULL REFERENCES {libraries_table}(id),
                collection_key    TEXT NOT NULL,
                name              TEXT,
                parent_key        TEXT,
                UNIQUE(library_pk, collection_key)
            );
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {items_table} (
                id            SERIAL PRIMARY KEY,
                library_pk    INTEGER NOT NULL REFERENCES {libraries_table}(id),
                zotero_key    TEXT NOT NULL,
                title         TEXT,
                authors       TEXT,
                abstract      TEXT,
                doi           TEXT,
                date          TEXT,
                url           TEXT,
                item_type     TEXT,
                updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(library_pk, zotero_key)
            );
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {item_collections_table} (
                item_id       INTEGER NOT NULL REFERENCES {items_table}(id),
                collection_id INTEGER NOT NULL REFERENCES {collections_table}(id),
                PRIMARY KEY (item_id, collection_id)
            );
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {tags_table} (
                id         SERIAL PRIMARY KEY,
                item_id    INTEGER NOT NULL REFERENCES {items_table}(id),
                tag        TEXT NOT NULL
            );
            """
        )
    else:
        cur.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS {libraries_table} (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                library_id        TEXT NOT NULL,
                library_type      TEXT NOT NULL,
                last_synced_at    TEXT,
                UNIQUE(library_id, library_type)
            );

            CREATE TABLE IF NOT EXISTS {collections_table} (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                library_pk        INTEGER NOT NULL,
                collection_key    TEXT NOT NULL,
                name              TEXT,
                parent_key        TEXT,
                UNIQUE(library_pk, collection_key),
                FOREIGN KEY (library_pk) REFERENCES {libraries_table}(id)
            );

            CREATE TABLE IF NOT EXISTS {items_table} (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                library_pk    INTEGER NOT NULL,
                zotero_key    TEXT NOT NULL,
                title         TEXT,
                authors       TEXT,
                abstract      TEXT,
                doi           TEXT,
                date          TEXT,
                url           TEXT,
                item_type     TEXT,
                updated_at    TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(library_pk, zotero_key),
                FOREIGN KEY (library_pk) REFERENCES {libraries_table}(id)
            );

            CREATE TABLE IF NOT EXISTS {item_collections_table} (
                item_id       INTEGER NOT NULL,
                collection_id INTEGER NOT NULL,
                PRIMARY KEY (item_id, collection_id),
                FOREIGN KEY (item_id) REFERENCES {items_table}(id),
                FOREIGN KEY (collection_id) REFERENCES {collections_table}(id)
            );

            CREATE TABLE IF NOT EXISTS {tags_table} (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id    INTEGER NOT NULL,
                tag        TEXT NOT NULL,
                FOREIGN KEY (item_id) REFERENCES {items_table}(id)
            );
            """
        )
    conn.commit()
    conn.close()


def get_or_create_library(library_id: str, library_type: str = "user") -> int:
    """Get or create a library row, return library PK."""
    conn = get_conn()
    cur = conn.cursor()
    table = qualify_table(ZOTERO_SCHEMA, "libraries")
    ph = placeholder()

    cur.execute(
        f"""
        SELECT id FROM {table}
        WHERE library_id = {ph} AND library_type = {ph}
        """,
        (library_id, library_type),
    )
    row = cur.fetchone()

    if row:
        library_pk = row["id"]
    else:
        params = placeholders(2)
        if is_postgres():
            cur.execute(
                f"""
                INSERT INTO {table} (library_id, library_type)
                VALUES ({params})
                RETURNING id
                """,
                (library_id, library_type),
            )
            library_pk = cur.fetchone()["id"]
        else:
            cur.execute(
                f"""
                INSERT INTO {table} (library_id, library_type)
                VALUES ({params})
                """,
                (library_id, library_type),
            )
            library_pk = cur.lastrowid
        conn.commit()

    conn.close()
    return library_pk


def upsert_collections(
    library_pk: int, collections: List[Dict[str, Any]]
) -> Dict[str, int]:
    """
    Insert or update collections from Zotero API response.

    Returns mapping of collection_key -> collection_id.
    """
    conn = get_conn()
    cur = conn.cursor()
    table = qualify_table(ZOTERO_SCHEMA, "collections")
    ph = placeholder()

    key_to_id: Dict[str, int] = {}

    for coll in collections:
        data = coll.get("data", {})
        collection_key = data.get("key")
        name = data.get("name")
        parent_key = data.get("parentCollection") or None

        if not collection_key:
            continue

        # Check if exists
        cur.execute(
            f"SELECT id FROM {table} WHERE library_pk = {ph} AND collection_key = {ph}",
            (library_pk, collection_key),
        )
        row = cur.fetchone()

        if row:
            # Update existing
            cur.execute(
                f"UPDATE {table} SET name = {ph}, parent_key = {ph} WHERE id = {ph}",
                (name, parent_key, row["id"]),
            )
            key_to_id[collection_key] = row["id"]
        else:
            # Insert new
            params = placeholders(4)
            if is_postgres():
                cur.execute(
                    f"""
                    INSERT INTO {table} (library_pk, collection_key, name, parent_key)
                    VALUES ({params})
                    RETURNING id
                    """,
                    (library_pk, collection_key, name, parent_key),
                )
                key_to_id[collection_key] = cur.fetchone()["id"]
            else:
                cur.execute(
                    f"""
                    INSERT INTO {table} (library_pk, collection_key, name, parent_key)
                    VALUES ({params})
                    """,
                    (library_pk, collection_key, name, parent_key),
                )
                key_to_id[collection_key] = cur.lastrowid

    conn.commit()
    conn.close()
    return key_to_id


def upsert_item(
    library_pk: int,
    item_data: Dict[str, Any],
    collection_key_to_id: Dict[str, int],
) -> Optional[int]:
    """
    Insert or update an item from Zotero.

    Args:
        library_pk: Library primary key
        item_data: Parsed item data from zotero_item_to_row()
        collection_key_to_id: Mapping of collection keys to IDs

    Returns:
        Item ID if inserted/updated, None if skipped (attachment).
    """
    from .api import zotero_item_to_row

    conn = get_conn()
    cur = conn.cursor()
    items_table = qualify_table(ZOTERO_SCHEMA, "items")
    tags_table = qualify_table(ZOTERO_SCHEMA, "tags")
    item_collections_table = qualify_table(ZOTERO_SCHEMA, "item_collections")
    ph = placeholder()

    # Parse item
    row = zotero_item_to_row(item_data)

    # Skip attachments
    if row["item_type"] == "attachment":
        conn.close()
        return None

    zotero_key = row["zotero_key"]
    tags = row.pop("tags")
    collection_keys = row.pop("collections")

    # Check if item exists
    cur.execute(
        f"SELECT id FROM {items_table} WHERE library_pk = {ph} AND zotero_key = {ph}",
        (library_pk, zotero_key),
    )
    existing = cur.fetchone()

    if existing:
        item_id = existing["id"]
        # Update existing item
        cur.execute(
            f"""
            UPDATE {items_table}
            SET title = {ph}, authors = {ph}, abstract = {ph}, doi = {ph},
                date = {ph}, url = {ph}, item_type = {ph}, updated_at = CURRENT_TIMESTAMP
            WHERE id = {ph}
            """,
            (
                row["title"],
                row["authors"],
                row["abstract"],
                row["doi"],
                row["date"],
                row["url"],
                row["item_type"],
                item_id,
            ),
        )
        # Delete old tags
        cur.execute(f"DELETE FROM {tags_table} WHERE item_id = {ph}", (item_id,))
        # Delete old collection links
        cur.execute(
            f"DELETE FROM {item_collections_table} WHERE item_id = {ph}", (item_id,)
        )
    else:
        # Insert new item
        params = placeholders(9)
        if is_postgres():
            cur.execute(
                f"""
                INSERT INTO {items_table} (
                    library_pk, zotero_key, title, authors, abstract,
                    doi, date, url, item_type
                )
                VALUES ({params})
                RETURNING id
                """,
                (
                    library_pk,
                    zotero_key,
                    row["title"],
                    row["authors"],
                    row["abstract"],
                    row["doi"],
                    row["date"],
                    row["url"],
                    row["item_type"],
                ),
            )
            item_id = cur.fetchone()["id"]
        else:
            cur.execute(
                f"""
                INSERT INTO {items_table} (
                    library_pk, zotero_key, title, authors, abstract,
                    doi, date, url, item_type
                )
                VALUES ({params})
                """,
                (
                    library_pk,
                    zotero_key,
                    row["title"],
                    row["authors"],
                    row["abstract"],
                    row["doi"],
                    row["date"],
                    row["url"],
                    row["item_type"],
                ),
            )
            item_id = cur.lastrowid

    # Insert tags
    for t in tags:
        cur.execute(
            f"INSERT INTO {tags_table} (item_id, tag) VALUES ({ph}, {ph})",
            (item_id, t),
        )

    # Insert collection links
    for coll_key in collection_keys:
        coll_id = collection_key_to_id.get(coll_key)
        if coll_id:
            cur.execute(
                f"INSERT INTO {item_collections_table} (item_id, collection_id) VALUES ({ph}, {ph})",
                (item_id, coll_id),
            )

    conn.commit()
    conn.close()
    return item_id


def get_collection_by_name_or_key(
    library_pk: int, name_or_key: str
) -> Optional[Dict[str, Any]]:
    """Get collection by name or key."""
    conn = get_conn()
    cur = conn.cursor()
    table = qualify_table(ZOTERO_SCHEMA, "collections")
    ph = placeholder()

    # Try by key first
    cur.execute(
        f"SELECT * FROM {table} WHERE library_pk = {ph} AND collection_key = {ph}",
        (library_pk, name_or_key),
    )
    row = cur.fetchone()

    if not row:
        # Try by name (case-insensitive)
        if is_postgres():
            cur.execute(
                f"SELECT * FROM {table} WHERE library_pk = {ph} AND LOWER(name) = LOWER({ph})",
                (library_pk, name_or_key),
            )
        else:
            cur.execute(
                f"SELECT * FROM {table} WHERE library_pk = {ph} AND LOWER(name) = LOWER({ph})",
                (library_pk, name_or_key),
            )
        row = cur.fetchone()

    conn.close()
    return dict(row) if row else None


def get_items_for_collection(collection_id: int) -> List[Dict[str, Any]]:
    """Get all items in a collection."""
    conn = get_conn()
    cur = conn.cursor()
    items_table = qualify_table(ZOTERO_SCHEMA, "items")
    item_collections_table = qualify_table(ZOTERO_SCHEMA, "item_collections")
    tags_table = qualify_table(ZOTERO_SCHEMA, "tags")
    ph = placeholder()

    cur.execute(
        f"""
        SELECT i.* FROM {items_table} i
        JOIN {item_collections_table} ic ON i.id = ic.item_id
        WHERE ic.collection_id = {ph}
        """,
        (collection_id,),
    )
    items = [dict(row) for row in cur.fetchall()]

    # Fetch tags for each item
    for item in items:
        cur.execute(
            f"SELECT tag FROM {tags_table} WHERE item_id = {ph}", (item["id"],)
        )
        item["tags"] = [r["tag"] for r in cur.fetchall()]

    conn.close()
    return items


def get_all_items(library_pk: int) -> List[Dict[str, Any]]:
    """Get all items in a library."""
    conn = get_conn()
    cur = conn.cursor()
    items_table = qualify_table(ZOTERO_SCHEMA, "items")
    tags_table = qualify_table(ZOTERO_SCHEMA, "tags")
    ph = placeholder()

    cur.execute(
        f"SELECT * FROM {items_table} WHERE library_pk = {ph}",
        (library_pk,),
    )
    items = [dict(row) for row in cur.fetchall()]

    # Fetch tags for each item
    for item in items:
        cur.execute(
            f"SELECT tag FROM {tags_table} WHERE item_id = {ph}", (item["id"],)
        )
        item["tags"] = [r["tag"] for r in cur.fetchall()]

    conn.close()
    return items


def get_all_collections(library_pk: int) -> List[Dict[str, Any]]:
    """Get all collections in a library."""
    conn = get_conn()
    cur = conn.cursor()
    table = qualify_table(ZOTERO_SCHEMA, "collections")
    item_collections_table = qualify_table(ZOTERO_SCHEMA, "item_collections")
    ph = placeholder()

    cur.execute(
        f"""
        SELECT c.*, COUNT(ic.item_id) as num_items
        FROM {table} c
        LEFT JOIN {item_collections_table} ic ON c.id = ic.collection_id
        WHERE c.library_pk = {ph}
        GROUP BY c.id
        ORDER BY c.name
        """,
        (library_pk,),
    )
    collections = [dict(row) for row in cur.fetchall()]

    conn.close()
    return collections


def update_library_sync_time(library_pk: int) -> None:
    """Update the last_synced_at timestamp for a library."""
    conn = get_conn()
    cur = conn.cursor()
    table = qualify_table(ZOTERO_SCHEMA, "libraries")
    ph = placeholder()

    cur.execute(
        f"UPDATE {table} SET last_synced_at = CURRENT_TIMESTAMP WHERE id = {ph}",
        (library_pk,),
    )
    conn.commit()
    conn.close()


def get_library(library_id: str, library_type: str = "user") -> Optional[Dict[str, Any]]:
    """Get library by library_id and library_type."""
    conn = get_conn()
    cur = conn.cursor()
    table = qualify_table(ZOTERO_SCHEMA, "libraries")
    ph = placeholder()

    cur.execute(
        f"SELECT * FROM {table} WHERE library_id = {ph} AND library_type = {ph}",
        (library_id, library_type),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None
