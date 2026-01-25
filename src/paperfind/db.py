"""Database helpers for SQLite and optional Postgres backend."""


import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Generator, Optional, Union

if TYPE_CHECKING:
    import psycopg

DBConnection = Union[sqlite3.Connection, "psycopg.Connection"]

from paperfind.config import DAILY_PAPERS_DB, ZOTERO_DB

DAILY_SCHEMA = "daily"
ZOTERO_SCHEMA = "zotero"


def db_url() -> Optional[str]:
    """Return the Postgres connection string if configured."""
    return os.getenv("PAPERFIND_DB_URL")


def is_postgres() -> bool:
    """Return True if Postgres is configured."""
    return bool(db_url())


def get_conn(schema: str) -> DBConnection:
    """Get a database connection for the requested schema."""
    if is_postgres():
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise ImportError(
                "Postgres support requires psycopg. Install with: pip install paperfind[postgres]"
            ) from exc

        return psycopg.connect(db_url(), row_factory=dict_row)

    if schema == DAILY_SCHEMA:
        path = DAILY_PAPERS_DB
    elif schema == ZOTERO_SCHEMA:
        path = ZOTERO_DB
    else:
        raise ValueError(f"Unknown schema '{schema}' for SQLite backend")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db(schema: str) -> Generator[DBConnection, None, None]:
    """Context manager for database connections.

    Automatically closes the connection when exiting the context,
    even if an exception occurs.

    Usage:
        with get_db(DAILY_SCHEMA) as conn:
            cur = conn.cursor()
            cur.execute(...)
    """
    conn = get_conn(schema)
    try:
        yield conn
    finally:
        conn.close()


def qualify_table(schema: str, table: str) -> str:
    """Return a schema-qualified table name for Postgres, or bare table name for SQLite."""
    return f"{schema}.{table}" if is_postgres() else table


def placeholder() -> str:
    """Return the parameter placeholder for the active backend."""
    return "%s" if is_postgres() else "?"


def placeholders(count: int) -> str:
    """Return a comma-separated placeholder list."""
    return ", ".join([placeholder()] * count)


def table_exists(conn: DBConnection, schema: str, table: str) -> bool:
    """Return True if the table exists."""
    cur = conn.cursor()
    if is_postgres():
        cur.execute("SELECT to_regclass(%s)", (f"{schema}.{table}",))
        row = cur.fetchone()
        return bool(row and row["to_regclass"])

    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
        (table,),
    )
    return cur.fetchone() is not None
