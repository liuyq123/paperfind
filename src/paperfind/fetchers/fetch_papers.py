"""
fetch_papers.py

Unified script to fetch papers from all sources: CrossRef, bioRxiv, medRxiv, arXiv, and ChemRxiv.

Usage:
    paperfind fetch                    # Fetch from all sources (today)
    paperfind fetch --days 7           # Fetch last 7 days
    paperfind fetch --source arxiv     # Fetch from specific source
    paperfind fetch --rebuild-vectors  # Also rebuild vector embeddings
"""

import time
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from paperfind.config import (
    BIORXIV_CATEGORIES,
    CHEMRXIV_CATEGORIES,
    DAILY_PAPERS_DB,
    MEDRXIV_CATEGORIES,
)
from paperfind.fetchers.db import init_db, upsert_work
from paperfind.fetchers.sources.arxiv import ARXIV_CATEGORIES, fetch_arxiv
from paperfind.fetchers.sources.biorxiv import fetch_biorxiv
from paperfind.fetchers.sources.chemrxiv import fetch_chemrxiv
from paperfind.fetchers.sources.crossref import fetch_crossref
from paperfind.fetchers.vector import rebuild_vectors, upsert_vectors_for_dois
from paperfind.logging import get_logger
from paperfind.types import PaperDict

logger = get_logger(__name__)


# ============ Helpers ============


def _fetch_rxiv_papers(
    start_date: date,
    end_date: date,
    server: str,
    categories: List[str],
) -> List[PaperDict]:
    """Fetch papers from bioRxiv or medRxiv.

    Args:
        start_date: Start of date range
        end_date: End of date range
        server: "biorxiv" or "medrxiv"
        categories: List of categories to fetch, or empty for all

    Returns:
        List of paper dictionaries
    """
    papers: List[PaperDict] = []

    if categories:
        for category in categories:
            fetched = fetch_biorxiv(start_date, end_date, server, category=category)
            papers.extend(fetched)
            logger.debug(f"    {category}: {len(fetched)} papers")
    else:
        logger.debug("    (all categories)")
        papers = fetch_biorxiv(start_date, end_date, server)

    return papers


# ============ Main ============


def fetch_all(
    days: int = 1,
    sources: Optional[List[str]] = None,
    arxiv_days: Optional[int] = None,
) -> Tuple[Dict[str, int], List[str]]:
    """
    Fetch papers from all sources.

    Args:
        days: Number of days to fetch
        sources: List of sources to fetch from (default: all)
        arxiv_days: Number of days to fetch for arXiv (default: same as days).
                    Useful since arXiv has batch processing delays.

    Returns:
        Tuple of (counts by source, list of fetched DOIs)

    Note:
        Categories are configured via environment variables:
        ARXIV_CATEGORIES, BIORXIV_CATEGORIES (see .env.example)
    """
    if sources is None:
        sources = ["crossref", "biorxiv", "medrxiv", "arxiv", "chemrxiv"]
    if days < 1:
        raise ValueError("days must be >= 1")

    conn = init_db()
    end_date = date.today()
    start_date = end_date - timedelta(days=days - 1)
    counts: Dict[str, int] = {}
    fetched_dois: List[str] = []

    # CrossRef
    if "crossref" in sources:
        logger.info("[CrossRef] Fetching journal articles and preprints...")
        crossref_papers = []

        for d in range(days):
            target = end_date - timedelta(days=d)
            for type_filter in ["journal-article", "posted-content"]:
                papers = fetch_crossref(target, type_filter)
                crossref_papers.extend(papers)

        for paper in crossref_papers:
            upsert_work(conn, paper)
            fetched_dois.append(paper["doi"])
        conn.commit()

        counts["crossref"] = len(crossref_papers)
        logger.info(f"    Stored {len(crossref_papers)} papers")

    # bioRxiv
    if "biorxiv" in sources:
        logger.info("[bioRxiv] Fetching preprints...")
        biorxiv_papers = _fetch_rxiv_papers(start_date, end_date, "biorxiv", BIORXIV_CATEGORIES)

        for paper in biorxiv_papers:
            upsert_work(conn, paper)
            fetched_dois.append(paper["doi"])
        conn.commit()

        counts["biorxiv"] = len(biorxiv_papers)
        logger.info(f"    Total: {len(biorxiv_papers)} preprints")

    # medRxiv
    if "medrxiv" in sources:
        logger.info("[medRxiv] Fetching preprints...")
        medrxiv_papers = _fetch_rxiv_papers(start_date, end_date, "medrxiv", MEDRXIV_CATEGORIES)

        for paper in medrxiv_papers:
            upsert_work(conn, paper)
            fetched_dois.append(paper["doi"])
        conn.commit()

        counts["medrxiv"] = len(medrxiv_papers)
        logger.info(f"    Total: {len(medrxiv_papers)} preprints")

    # arXiv
    if "arxiv" in sources:
        arxiv_days_to_use = arxiv_days if arxiv_days is not None else days
        logger.info(f"[arXiv] Fetching preprints (last {arxiv_days_to_use} days)...")
        arxiv_papers = []

        for category in ARXIV_CATEGORIES:
            papers = fetch_arxiv(category, days=arxiv_days_to_use)
            arxiv_papers.extend(papers)
            logger.debug(f"    {category}: {len(papers)} papers")
            time.sleep(1)  # Rate limiting

        for paper in arxiv_papers:
            upsert_work(conn, paper)
            fetched_dois.append(paper["doi"])
        conn.commit()

        counts["arxiv"] = len(arxiv_papers)
        logger.info(f"    Total: {len(arxiv_papers)} preprints")

    # ChemRxiv
    if "chemrxiv" in sources:
        logger.info("[ChemRxiv] Fetching preprints...")
        chemrxiv_papers = fetch_chemrxiv(start_date, end_date, categories=CHEMRXIV_CATEGORIES)

        for paper in chemrxiv_papers:
            upsert_work(conn, paper)
            fetched_dois.append(paper["doi"])
        conn.commit()

        counts["chemrxiv"] = len(chemrxiv_papers)
        logger.info(f"    Total: {len(chemrxiv_papers)} preprints")

    conn.close()
    return counts, fetched_dois


def run_fetch(
    days: int = 1,
    arxiv_days: Optional[int] = None,
    sources: Optional[List[str]] = None,
    rebuild_vectors_flag: bool = False,
    vectors_only: bool = False,
) -> None:
    """Run paper fetching with parsed parameters."""
    if vectors_only:
        rebuild_vectors()
        return
    if days < 1:
        logger.error("Days must be >= 1.")
        return

    logger.info("=" * 50)
    logger.info(f"Fetching papers from last {days} day(s)")
    logger.info("=" * 50)

    counts, fetched_dois = fetch_all(
        days=days,
        sources=sources,
        arxiv_days=arxiv_days,
    )

    logger.info("=" * 50)
    logger.info("Summary:")
    total = 0
    for source, count in counts.items():
        logger.info(f"  {source}: {count} papers")
        total += count
    logger.info(f"  Total: {total} papers stored in {DAILY_PAPERS_DB}")
    logger.info("=" * 50)

    if rebuild_vectors_flag:
        rebuild_vectors()
    else:
        upsert_vectors_for_dois(sorted(set(fetched_dois)))
