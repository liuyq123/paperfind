"""Database helpers for paper fetchers."""

from paperfind.db import DAILY_SCHEMA, get_conn, is_postgres, placeholders, qualify_table
from paperfind.types import PaperDict


def init_db():
    """Initialize database with works table."""
    conn = get_conn(DAILY_SCHEMA)
    cur = conn.cursor()
    table = qualify_table(DAILY_SCHEMA, "works")

    if is_postgres():
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {DAILY_SCHEMA}")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                doi TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT,
                abstract TEXT,
                created_date DATE,
                type TEXT,
                source TEXT
            );
            """
        )
    else:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                doi TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT,
                abstract TEXT,
                created_date DATE,
                type TEXT,
                source TEXT
            );
            """
        )

    conn.commit()
    return conn


def upsert_work(conn, work: PaperDict) -> None:
    """Insert or update a work in the database."""
    cur = conn.cursor()
    table = qualify_table(DAILY_SCHEMA, "works")
    params = placeholders(7)
    cur.execute(
        f"""
        INSERT INTO {table} (doi, title, authors, abstract, created_date, type, source)
        VALUES ({params})
        ON CONFLICT(doi) DO UPDATE SET
            title = excluded.title,
            authors = excluded.authors,
            abstract = excluded.abstract,
            created_date = excluded.created_date,
            type = excluded.type,
            source = excluded.source;
        """,
        (
            work["doi"],
            work["title"],
            work["authors"],
            work["abstract"],
            work["created_date"],
            work["type"],
            work["source"],
        ),
    )
