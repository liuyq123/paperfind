"""ChemRxiv fetcher."""

from datetime import date
from typing import Any, Dict, List, Optional

import requests

from paperfind.logging import get_logger
from paperfind.retry import retry_request
from paperfind.types import PaperDict

logger = get_logger(__name__)

CHEMRXIV_API = "https://chemrxiv.org/engage/chemrxiv/public-api/v1/items"
PAGE_SIZE = 50  # ChemRxiv API limit


def fetch_chemrxiv(
    start_date: date,
    end_date: date,
    categories: Optional[List[str]] = None,
) -> List[PaperDict]:
    """Fetch preprints from ChemRxiv.

    Args:
        start_date: Start of date range (inclusive)
        end_date: End of date range (inclusive)
        categories: List of category IDs to filter by. If empty, fetches all.
                    See https://chemrxiv.org/engage/chemrxiv/public-api/v1/categories

    Returns:
        List of paper dictionaries
    """
    papers: List[PaperDict] = []
    skip = 0

    while True:
        params: Dict[str, Any] = {"limit": PAGE_SIZE, "skip": skip}

        # Add category filter if specified
        if categories:
            params["categoryIds"] = ",".join(categories)

        try:
            resp = retry_request(
                lambda: requests.get(CHEMRXIV_API, params=params, timeout=60),
                description="ChemRxiv",
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching ChemRxiv: {e}")
            break

        items = data.get("itemHits", [])
        if not items:
            break

        reached_cutoff = False

        for hit in items:
            item = hit.get("item", {})

            # Parse published date
            pub_date_str = item.get("publishedDate", "")
            if not pub_date_str:
                continue

            try:
                paper_date = date.fromisoformat(pub_date_str[:10])
            except (ValueError, TypeError):
                continue

            # Skip if before date range
            if paper_date < start_date:
                reached_cutoff = True
                continue

            # Skip if after date range
            if paper_date > end_date:
                continue

            doi = item.get("doi")
            title = item.get("title", "")
            abstract = item.get("abstract", "")

            if not doi or not title or not abstract:
                continue

            # Format authors
            authors = []
            for author in item.get("authors", []):
                first = author.get("firstName", "")
                last = author.get("lastName", "")
                if first and last:
                    authors.append(f"{first} {last}")
                elif last:
                    authors.append(last)
            authors_str = ", ".join(authors)

            papers.append({
                "doi": doi,
                "title": title,
                "authors": authors_str,
                "abstract": abstract,
                "created_date": pub_date_str[:10],
                "type": "preprint",
                "source": "chemrxiv",
            })

        # Stop if we've reached papers older than our range
        if reached_cutoff or len(items) < PAGE_SIZE:
            break

        skip += PAGE_SIZE

    return papers
