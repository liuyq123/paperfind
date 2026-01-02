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

    projects_table = qualify_table(ZOTERO_SCHEMA, "projects")
    items_table = qualify_table(ZOTERO_SCHEMA, "items")
    tags_table = qualify_table(ZOTERO_SCHEMA, "tags")

    if is_postgres():
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {ZOTERO_SCHEMA}")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {projects_table} (
                id                SERIAL PRIMARY KEY,
                name              TEXT NOT NULL,
                library_id        TEXT NOT NULL,
                library_type      TEXT NOT NULL,
                collection_name   TEXT,
                collection_key    TEXT,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {items_table} (
                id            SERIAL PRIMARY KEY,
                project_id    INTEGER NOT NULL REFERENCES {projects_table}(id),
                zotero_key    TEXT NOT NULL,
                title         TEXT,
                authors       TEXT,
                abstract      TEXT,
                doi           TEXT,
                date          TEXT,
                url           TEXT,
                item_type     TEXT,
                updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project_id, zotero_key)
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
            CREATE TABLE IF NOT EXISTS {projects_table} (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                name              TEXT NOT NULL,
                library_id        TEXT NOT NULL,
                library_type      TEXT NOT NULL,
                collection_name   TEXT,
                collection_key    TEXT,
                created_at        TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS {items_table} (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id    INTEGER NOT NULL,
                zotero_key    TEXT NOT NULL,
                title         TEXT,
                authors       TEXT,
                abstract      TEXT,
                doi           TEXT,
                date          TEXT,
                url           TEXT,
                item_type     TEXT,
                updated_at    TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project_id, zotero_key),
                FOREIGN KEY (project_id) REFERENCES {projects_table}(id)
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


def get_or_create_project(
    name: str,
    library_id: str,
    library_type: str = "user",
    collection_name: Optional[str] = None,
    collection_key: Optional[str] = None,
) -> int:
    """Create or get a project row."""
    conn = get_conn()
    cur = conn.cursor()
    table = qualify_table(ZOTERO_SCHEMA, "projects")
    ph = placeholder()

    if collection_key is None:
        cur.execute(
            f"""
            SELECT id FROM {table}
            WHERE name = {ph} AND library_id = {ph} AND library_type = {ph} AND collection_key IS NULL
            """,
            (name, library_id, library_type),
        )
    else:
        cur.execute(
            f"""
            SELECT id FROM {table}
            WHERE name = {ph} AND library_id = {ph} AND library_type = {ph} AND collection_key = {ph}
            """,
            (name, library_id, library_type, collection_key),
        )

    row = cur.fetchone()
    if row:
        project_id = row["id"]
    else:
        params = placeholders(5)
        if is_postgres():
            cur.execute(
                f"""
                INSERT INTO {table} (name, library_id, library_type, collection_name, collection_key)
                VALUES ({params})
                RETURNING id
                """,
                (name, library_id, library_type, collection_name, collection_key),
            )
            project_id = cur.fetchone()["id"]
        else:
            cur.execute(
                f"""
                INSERT INTO {table} (name, library_id, library_type, collection_name, collection_key)
                VALUES ({params})
                """,
                (name, library_id, library_type, collection_name, collection_key),
            )
            project_id = cur.lastrowid
        conn.commit()

    conn.close()
    return project_id


def replace_project_items(project_id: int, items: List[ZoteroItem]) -> List[int]:
    """Replace all items for a project with fresh data from Zotero."""
    from .api import zotero_item_to_row

    conn = get_conn()
    cur = conn.cursor()
    items_table = qualify_table(ZOTERO_SCHEMA, "items")
    tags_table = qualify_table(ZOTERO_SCHEMA, "tags")
    ph = placeholder()

    # Delete previous items & tags for this project
    cur.execute(f"SELECT id FROM {items_table} WHERE project_id = {ph}", (project_id,))
    old_ids = [row["id"] for row in cur.fetchall()]
    if old_ids:
        placeholders_sql = placeholders(len(old_ids))
        cur.execute(
            f"DELETE FROM {tags_table} WHERE item_id IN ({placeholders_sql})",
            old_ids,
        )
        cur.execute(f"DELETE FROM {items_table} WHERE project_id = {ph}", (project_id,))

    # Insert new items
    new_item_ids: List[int] = []
    for it in items:
        data = it.get("data", {})
        item_type = data.get("itemType")

        # Skip attachments
        if item_type == "attachment":
            continue

        row = zotero_item_to_row(it)
        tags = row.pop("tags")

        params = placeholders(9)
        if is_postgres():
            cur.execute(
                f"""
                INSERT INTO {items_table} (
                    project_id,
                    zotero_key,
                    title,
                    authors,
                    abstract,
                    doi,
                    date,
                    url,
                    item_type
                )
                VALUES ({params})
                RETURNING id
                """,
                (
                    project_id,
                    row["zotero_key"],
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
                    project_id,
                    zotero_key,
                    title,
                    authors,
                    abstract,
                    doi,
                    date,
                    url,
                    item_type
                )
                VALUES ({params})
                """,
                (
                    project_id,
                    row["zotero_key"],
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
        new_item_ids.append(item_id)

        for t in tags:
            cur.execute(
                f"INSERT INTO {tags_table} (item_id, tag) VALUES ({ph}, {ph})",
                (item_id, t),
            )

    conn.commit()
    conn.close()
    return new_item_ids
