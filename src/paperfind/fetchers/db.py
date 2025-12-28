"""Database helpers for paper fetchers."""

from typing import Dict
import sqlite3

from paperfind.config import DAILY_PAPERS_DB


def init_db() -> sqlite3.Connection:
    """Initialize database with works table."""
    DAILY_PAPERS_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DAILY_PAPERS_DB)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS works (
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


def upsert_work(conn: sqlite3.Connection, work: Dict) -> None:
    """Insert or update a work in the database."""
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO works (doi, title, authors, abstract, created_date, type, source)
        VALUES (?, ?, ?, ?, ?, ?, ?)
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
