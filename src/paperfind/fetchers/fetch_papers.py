"""
fetch_papers.py

Unified script to fetch papers from all sources: CrossRef, bioRxiv, medRxiv, and arXiv.

Usage:
    paperfind fetch                    # Fetch from all sources (today)
    paperfind fetch --days 7           # Fetch last 7 days
    paperfind fetch --source arxiv     # Fetch from specific source
    paperfind fetch --rebuild-vectors  # Also rebuild vector embeddings
"""

from datetime import date, timedelta
import time
from typing import Dict, List, Optional

from paperfind.config import DAILY_PAPERS_DB
from paperfind.fetchers.db import init_db, upsert_work
from paperfind.fetchers.sources.arxiv import ARXIV_CATEGORIES, fetch_arxiv
from paperfind.fetchers.sources.biorxiv import BIORXIV_CATEGORIES, fetch_biorxiv
from paperfind.fetchers.sources.crossref import fetch_crossref
from paperfind.fetchers.vector import rebuild_vectors, upsert_vectors_for_dois


# ============ Main ============

def fetch_all(
    days: int = 1,
    sources: Optional[List[str]] = None,
    biorxiv_category: Optional[str] = None,
    medrxiv_category: Optional[str] = None,
) -> tuple[Dict[str, int], List[str]]:
    """
    Fetch papers from all sources.

    Args:
        days: Number of days to fetch
        sources: List of sources to fetch from (default: all)

    Returns:
        Tuple of (counts by source, list of fetched DOIs)
    """
    if sources is None:
        sources = ["crossref", "biorxiv", "medrxiv", "arxiv"]

    conn = init_db()
    end_date = date.today()
    start_date = end_date - timedelta(days=days - 1)
    counts: Dict[str, int] = {}
    fetched_dois: List[str] = []

    # CrossRef
    if "crossref" in sources:
        print("\n[CrossRef] Fetching journal articles and preprints...")
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
        print(f"    Stored {len(crossref_papers)} papers")

    # bioRxiv
    if "biorxiv" in sources:
        print("\n[bioRxiv] Fetching preprints...")
        papers = fetch_biorxiv(
            start_date,
            end_date,
            "biorxiv",
            category=biorxiv_category,
        )
        for paper in papers:
            upsert_work(conn, paper)
            fetched_dois.append(paper["doi"])
        conn.commit()

        counts["biorxiv"] = len(papers)
        print(f"    Stored {len(papers)} preprints")

    # medRxiv
    if "medrxiv" in sources:
        print("\n[medRxiv] Fetching preprints...")
        papers = fetch_biorxiv(
            start_date,
            end_date,
            "medrxiv",
            category=medrxiv_category,
        )
        for paper in papers:
            upsert_work(conn, paper)
            fetched_dois.append(paper["doi"])
        conn.commit()

        counts["medrxiv"] = len(papers)
        print(f"    Stored {len(papers)} preprints")

    # arXiv
    if "arxiv" in sources:
        print("\n[arXiv] Fetching preprints...")
        arxiv_papers = []

        for category in ARXIV_CATEGORIES:
            papers = fetch_arxiv(category, days=days)
            arxiv_papers.extend(papers)
            print(f"    {category}: {len(papers)} papers")
            time.sleep(1)  # Rate limiting

        for paper in arxiv_papers:
            upsert_work(conn, paper)
            fetched_dois.append(paper["doi"])
        conn.commit()

        counts["arxiv"] = len(arxiv_papers)
        print(f"    Total: {len(arxiv_papers)} preprints")

    conn.close()
    return counts, fetched_dois


def run_fetch(
    days: int = 1,
    sources: Optional[List[str]] = None,
    biorxiv_category: Optional[str] = None,
    medrxiv_category: Optional[str] = None,
    rebuild_vectors_flag: bool = False,
    vectors_only: bool = False,
) -> None:
    """Run paper fetching with parsed parameters."""
    if vectors_only:
        rebuild_vectors()
        return

    print(f"{'='*50}")
    print(f"Fetching papers from last {days} day(s)")
    print(f"{'='*50}")

    counts, fetched_dois = fetch_all(
        days=days,
        sources=sources,
        biorxiv_category=biorxiv_category,
        medrxiv_category=medrxiv_category,
    )

    print(f"\n{'='*50}")
    print("Summary:")
    total = 0
    for source, count in counts.items():
        print(f"  {source}: {count} papers")
        total += count
    print(f"  Total: {total} papers stored in {DAILY_PAPERS_DB}")
    print(f"{'='*50}")

    if rebuild_vectors_flag:
        rebuild_vectors()
    else:
        upsert_vectors_for_dois(sorted(set(fetched_dois)))
