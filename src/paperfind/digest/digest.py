"""
digest.py

Main digest command logic - fetches papers, generates recommendations, sends email.

Usage:
    paperfind digest                    # Fetch, recommend, and send email
    paperfind digest --dry-run          # Preview HTML without sending
    paperfind digest --days 7 -k 20     # Custom settings
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

from paperfind.config import EMAIL_FROM, EMAIL_TO, SMTP_PASSWORD, SMTP_USER
from paperfind.digest.email import send_email
from paperfind.digest.template import render_digest
from paperfind.fetchers.db import get_sent_dois, prune_sent_recommendations, record_sent_dois
from paperfind.fetchers.fetch_papers import fetch_all
from paperfind.fetchers.vector import upsert_vectors_for_dois
from paperfind.logging import get_logger
from paperfind.search.recommend import get_recommendations

logger = get_logger(__name__)


def run_digest(
    days: int = 1,
    arxiv_days: Optional[int] = None,
    num_recommendations: int = 10,
    collection: Optional[str] = None,
    dry_run: bool = False,
    skip_fetch: bool = False,
    rerank: bool = True,
    max_age_days: Optional[int] = None,
) -> None:
    """
    Run the full digest pipeline: fetch papers, generate recommendations, send email.

    Args:
        days: Number of days to fetch papers for
        arxiv_days: Number of days to fetch for arXiv (default: same as days).
                    Useful since arXiv has batch processing delays.
        num_recommendations: Number of recommendations to include
        collection: Optional Zotero collection to base recommendations on
        dry_run: If True, print HTML instead of sending email
        skip_fetch: If True, skip fetching and use existing papers
        rerank: If True, use cross-encoder reranking (default: True)
        max_age_days: Only recommend papers published within this many days
    """
    today = date.today()

    if not skip_fetch:
        if days < 1:
            logger.error("Days must be >= 1.")
            return
        # Step 1: Fetch new papers
        logger.info("=" * 50)
        logger.info(f"Step 1: Fetching papers from last {days} day(s)")
        logger.info("=" * 50)
        counts, fetched_dois = fetch_all(days=days, arxiv_days=arxiv_days)

        # Step 2: Update vector embeddings (incremental - skips existing)
        logger.info("=" * 50)
        logger.info("Step 2: Updating vector embeddings")
        logger.info("=" * 50)
        upsert_vectors_for_dois(sorted(set(fetched_dois)))
    else:
        logger.info("Skipping fetch, using existing papers...")

    # Step 3: Generate recommendations (excluding previously sent)
    logger.info("=" * 50)
    logger.info("Step 3: Generating recommendations")
    logger.info("=" * 50)

    # Get previously sent DOIs to avoid repeats
    sent_dois = get_sent_dois()
    if sent_dois:
        logger.info(f"    Excluding {len(sent_dois)} previously sent recommendations")

    recommendations, rerank_used = get_recommendations(
        k=num_recommendations,
        collection=collection,
        rerank=rerank,
        return_rerank_used=True,
        max_age_days=max_age_days,
        exclude_dois=sent_dois,
    )

    if not recommendations:
        logger.warning("No recommendations found. Make sure you have synced your Zotero library.")
        return

    logger.info(f"Generated {len(recommendations)} recommendations")

    # Step 4: Render HTML
    html = render_digest(recommendations, today, rerank=rerank_used)

    # Step 5: Send or print
    if dry_run:
        logger.info("=" * 50)
        logger.info("Dry run - HTML output:")
        logger.info("=" * 50)
        print(html)
    else:
        logger.info("=" * 50)
        logger.info("Step 4: Sending email")
        logger.info("=" * 50)

        if not EMAIL_TO:
            logger.error("EMAIL_TO not set in .env. Use --dry-run to preview without sending.")
            return
        if not SMTP_USER or not SMTP_PASSWORD:
            logger.error("SMTP_USER and SMTP_PASSWORD must be set in .env to send emails.")
            return
        if not EMAIL_FROM:
            logger.error("EMAIL_FROM must be set in .env to send emails.")
            return

        to_addresses = [addr.strip() for addr in EMAIL_TO.split(",") if addr.strip()]
        subject = f"Paper Recommendations - {today.strftime('%B %d, %Y')}"

        try:
            send_email(subject=subject, html_body=html, to_addresses=to_addresses)
        except ValueError as exc:
            logger.error(f"Email sending failed: {exc}")
            return

        # Record sent DOIs to avoid repeats in future digests
        sent_paper_dois = [doi for doi, _ in recommendations]
        record_sent_dois(sent_paper_dois)
        logger.info(f"    Recorded {len(sent_paper_dois)} recommendations as sent")

        # Prune old sent records (allow papers to resurface after 30 days)
        prune_cutoff = date.today() - timedelta(days=30)
        prune_sent_recommendations(prune_cutoff)

    logger.info("Digest complete!")
