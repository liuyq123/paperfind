"""
digest.py

Main digest command logic - fetches papers, generates recommendations, sends email.

Usage:
    paperfind digest                    # Fetch, recommend, and send email
    paperfind digest --dry-run          # Preview HTML without sending
    paperfind digest --days 7 -k 20     # Custom settings
"""

from datetime import date
from typing import Optional

from paperfind.config import EMAIL_TO
from paperfind.digest.email import send_email
from paperfind.digest.template import render_digest
from paperfind.fetchers.fetch_papers import fetch_all
from paperfind.fetchers.vector import rebuild_vectors
from paperfind.search.recommend import get_recommendations


def run_digest(
    days: int = 1,
    num_recommendations: int = 10,
    collection: Optional[str] = None,
    dry_run: bool = False,
    skip_fetch: bool = False,
) -> None:
    """
    Run the full digest pipeline: fetch papers, generate recommendations, send email.

    Args:
        days: Number of days to fetch papers for
        num_recommendations: Number of recommendations to include
        collection: Optional Zotero collection to base recommendations on
        dry_run: If True, print HTML instead of sending email
        skip_fetch: If True, skip fetching and use existing papers
    """
    today = date.today()

    if not skip_fetch:
        # Step 1: Fetch new papers
        print(f"\n{'='*50}")
        print(f"Step 1: Fetching papers from last {days} day(s)")
        print(f"{'='*50}")
        fetch_all(days=days)

        # Step 2: Rebuild vector embeddings
        print(f"\n{'='*50}")
        print("Step 2: Rebuilding vector embeddings")
        print(f"{'='*50}")
        rebuild_vectors()
    else:
        print("Skipping fetch, using existing papers...")

    # Step 3: Generate recommendations
    print(f"\n{'='*50}")
    print("Step 3: Generating recommendations")
    print(f"{'='*50}")
    recommendations = get_recommendations(k=num_recommendations, collection=collection)

    if not recommendations:
        print("No recommendations found. Make sure you have synced your Zotero library.")
        return

    print(f"Generated {len(recommendations)} recommendations")

    # Step 4: Render HTML
    html = render_digest(recommendations, today)

    # Step 5: Send or print
    if dry_run:
        print(f"\n{'='*50}")
        print("Dry run - HTML output:")
        print(f"{'='*50}\n")
        print(html)
    else:
        print(f"\n{'='*50}")
        print("Step 4: Sending email")
        print(f"{'='*50}")

        if not EMAIL_TO:
            print("Error: EMAIL_TO not set in .env")
            print("Use --dry-run to preview without sending")
            return

        to_addresses = [addr.strip() for addr in EMAIL_TO.split(",") if addr.strip()]
        subject = f"Paper Recommendations - {today.strftime('%B %d, %Y')}"

        send_email(subject=subject, html_body=html, to_addresses=to_addresses)

    print("\nDigest complete!")
