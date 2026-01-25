"""Database helpers for paper fetchers."""


from datetime import date

from paperfind.db import (
    DAILY_SCHEMA,
    DBConnection,
    get_conn,
    get_db,
    is_postgres,
    placeholder,
    placeholders,
    qualify_table,
    table_exists,
)
from paperfind.logging import get_logger
from paperfind.types import PaperDict

logger = get_logger(__name__)


def init_db() -> DBConnection:
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


def upsert_work(conn: DBConnection, work: PaperDict) -> None:
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


def get_old_dois(cutoff_date: date) -> list[str]:
    """Get DOIs for papers older than the cutoff date.

    Args:
        cutoff_date: Papers created before this date will be returned.

    Returns:
        List of DOIs for old papers.
    """
    with get_db(DAILY_SCHEMA) as conn:
        cur = conn.cursor()
        table = qualify_table(DAILY_SCHEMA, "works")
        ph = placeholder()

        cur.execute(
            f"SELECT doi FROM {table} WHERE created_date < {ph}",
            (cutoff_date,),
        )
        return [row["doi"] for row in cur.fetchall()]


def prune_papers(cutoff_date: date) -> tuple[int, list[str]]:
    """Delete papers older than the cutoff date.

    Args:
        cutoff_date: Papers created before this date will be deleted.

    Returns:
        Tuple of (number of papers deleted, list of deleted DOIs).
    """
    # First get the DOIs to delete (for vector cleanup)
    dois = get_old_dois(cutoff_date)

    if not dois:
        return 0, []

    with get_db(DAILY_SCHEMA) as conn:
        cur = conn.cursor()
        table = qualify_table(DAILY_SCHEMA, "works")
        ph = placeholder()

        cur.execute(
            f"DELETE FROM {table} WHERE created_date < {ph}",
            (cutoff_date,),
        )
        deleted = cur.rowcount
        conn.commit()

    logger.info(f"Deleted {deleted} papers from database")
    return deleted, dois


def init_sent_recommendations_table() -> None:
    """Create the sent_recommendations table if it doesn't exist."""
    with get_db(DAILY_SCHEMA) as conn:
        cur = conn.cursor()
        table = qualify_table(DAILY_SCHEMA, "sent_recommendations")

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                doi TEXT PRIMARY KEY,
                sent_date DATE NOT NULL
            );
            """
        )
        conn.commit()


def get_sent_dois() -> set[str]:
    """Get DOIs of previously sent recommendations.

    Returns:
        Set of DOIs that have been sent in previous digests.
    """
    with get_db(DAILY_SCHEMA) as conn:
        if not table_exists(conn, DAILY_SCHEMA, "sent_recommendations"):
            return set()

        cur = conn.cursor()
        table = qualify_table(DAILY_SCHEMA, "sent_recommendations")

        cur.execute(f"SELECT doi FROM {table}")
        return {row["doi"] for row in cur.fetchall()}


def get_sent_dois_from_last_n_digests(n: int) -> set[str]:
    """Get DOIs from the last N digests (by distinct sent_date).

    Args:
        n: Number of most recent digests to include.

    Returns:
        Set of DOIs that were sent in the last N digests.
    """
    if n < 1:
        return set()

    with get_db(DAILY_SCHEMA) as conn:
        if not table_exists(conn, DAILY_SCHEMA, "sent_recommendations"):
            return set()

        cur = conn.cursor()
        table = qualify_table(DAILY_SCHEMA, "sent_recommendations")

        # Get the last N distinct sent_dates
        cur.execute(f"SELECT DISTINCT sent_date FROM {table} ORDER BY sent_date DESC LIMIT {n}")
        dates = [row["sent_date"] for row in cur.fetchall()]

        if not dates:
            return set()

        # Get all DOIs from those dates
        placeholders_str = ", ".join([placeholder() for _ in dates])
        cur.execute(f"SELECT doi FROM {table} WHERE sent_date IN ({placeholders_str})", tuple(dates))
        return {row["doi"] for row in cur.fetchall()}


def record_sent_dois(dois: list[str]) -> int:
    """Record DOIs as sent in the sent_recommendations table.

    Args:
        dois: List of DOIs to record as sent.

    Returns:
        Number of DOIs recorded.
    """
    if not dois:
        return 0

    init_sent_recommendations_table()

    with get_db(DAILY_SCHEMA) as conn:
        cur = conn.cursor()
        table = qualify_table(DAILY_SCHEMA, "sent_recommendations")
        params = placeholders(2)

        today = date.today()
        recorded = 0
        for doi in dois:
            cur.execute(
                f"""
                INSERT INTO {table} (doi, sent_date)
                VALUES ({params})
                ON CONFLICT(doi) DO UPDATE SET sent_date = excluded.sent_date;
                """,
                (doi, today),
            )
            recorded += 1

        conn.commit()

    logger.debug(f"Recorded {recorded} sent recommendations")
    return recorded


def prune_sent_recommendations(cutoff_date: date) -> int:
    """Delete sent recommendations older than the cutoff date.

    This allows papers to resurface in recommendations after some time.

    Args:
        cutoff_date: Sent recommendations before this date will be deleted.

    Returns:
        Number of records deleted.
    """
    with get_db(DAILY_SCHEMA) as conn:
        if not table_exists(conn, DAILY_SCHEMA, "sent_recommendations"):
            return 0

        cur = conn.cursor()
        table = qualify_table(DAILY_SCHEMA, "sent_recommendations")
        ph = placeholder()

        cur.execute(
            f"DELETE FROM {table} WHERE sent_date < {ph}",
            (cutoff_date,),
        )
        deleted = cur.rowcount
        conn.commit()

    if deleted > 0:
        logger.info(f"Pruned {deleted} old sent recommendations")
    return deleted


def get_paper_by_doi(doi: str) -> PaperDict | None:
    """Get paper details by DOI from the works table.

    Args:
        doi: Paper DOI to look up.

    Returns:
        Paper dict if found, None otherwise.
    """
    with get_db(DAILY_SCHEMA) as conn:
        if not table_exists(conn, DAILY_SCHEMA, "works"):
            return None

        cur = conn.cursor()
        table = qualify_table(DAILY_SCHEMA, "works")
        ph = placeholder()

        cur.execute(
            f"SELECT * FROM {table} WHERE doi = {ph}",
            (doi,),
        )
        row = cur.fetchone()
        return dict(row) if row else None
