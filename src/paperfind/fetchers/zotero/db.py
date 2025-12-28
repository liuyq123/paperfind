"""Database functions for Zotero sync."""

import sqlite3
from typing import Dict, List, Optional

from paperfind.config import ZOTERO_DB


def get_conn() -> sqlite3.Connection:
    """Get database connection."""
    ZOTERO_DB.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(ZOTERO_DB)


def init_db() -> None:
    """Initialize the database schema."""
    conn = get_conn()
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS projects (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            name              TEXT NOT NULL,
            library_id        TEXT NOT NULL,
            library_type      TEXT NOT NULL,
            collection_name   TEXT,
            collection_key    TEXT,
            created_at        TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS items (
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
            FOREIGN KEY (project_id) REFERENCES projects(id)
        );

        CREATE TABLE IF NOT EXISTS tags (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            item_id    INTEGER NOT NULL,
            tag        TEXT NOT NULL,
            FOREIGN KEY (item_id) REFERENCES items(id)
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

    if collection_key is None:
        cur.execute(
            """
            SELECT id FROM projects
            WHERE name = ? AND library_id = ? AND library_type = ? AND collection_key IS NULL
            """,
            (name, library_id, library_type),
        )
    else:
        cur.execute(
            """
            SELECT id FROM projects
            WHERE name = ? AND library_id = ? AND library_type = ? AND collection_key = ?
            """,
            (name, library_id, library_type, collection_key),
        )

    row = cur.fetchone()
    if row:
        project_id = row[0]
    else:
        cur.execute(
            """
            INSERT INTO projects (name, library_id, library_type, collection_name, collection_key)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, library_id, library_type, collection_name, collection_key),
        )
        project_id = cur.lastrowid
        conn.commit()

    conn.close()
    return project_id


def replace_project_items(project_id: int, items: List[Dict]) -> List[int]:
    """Replace all items for a project with fresh data from Zotero."""
    from .api import zotero_item_to_row

    conn = get_conn()
    cur = conn.cursor()

    # Delete previous items & tags for this project
    cur.execute("SELECT id FROM items WHERE project_id = ?", (project_id,))
    old_ids = [row[0] for row in cur.fetchall()]
    if old_ids:
        cur.execute(
            f"DELETE FROM tags WHERE item_id IN ({','.join('?' for _ in old_ids)})",
            old_ids,
        )
        cur.execute("DELETE FROM items WHERE project_id = ?", (project_id,))

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

        cur.execute(
            """
            INSERT INTO items (project_id, zotero_key, title, authors, abstract, doi, date, url, item_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                "INSERT INTO tags (item_id, tag) VALUES (?, ?)",
                (item_id, t),
            )

    conn.commit()
    conn.close()
    return new_item_ids
