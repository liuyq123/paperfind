"""bioRxiv/medRxiv fetcher."""

from datetime import date
from typing import List, Optional

import requests

from paperfind.config import BIORXIV_CATEGORIES
from paperfind.logging import get_logger
from paperfind.types import PaperDict

logger = get_logger(__name__)

BIORXIV_API = "https://api.biorxiv.org/details"


def fetch_biorxiv(
    start_date: date,
    end_date: date,
    server: str = "biorxiv",
    category: Optional[str] = None,
) -> List[PaperDict]:
    """Fetch preprints from bioRxiv or medRxiv.

    Args:
        start_date: Start of date range
        end_date: End of date range
        server: "biorxiv" or "medrxiv"
        category: Optional category filter (e.g., "bioinformatics")

    Returns:
        List of paper dictionaries
    """
    papers: List[PaperDict] = []
    cursor = 0
    interval = f"{start_date}/{end_date}"

    while True:
        url = f"{BIORXIV_API}/{server}/{interval}/{cursor}/json"

        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching {server}: {e}")
            break

        collection = data.get("collection", [])
        if not collection:
            break

        for item in collection:
            if category:
                item_category = item.get("category", "").lower()
                # Normalize hyphens to spaces for comparison (config uses hyphens, API uses spaces)
                normalized_category = category.lower().replace("-", " ")
                if normalized_category not in item_category:
                    continue

            doi = item.get("doi")
            title = item.get("title", "")
            abstract = item.get("abstract", "")

            if not doi or not title or not abstract:
                continue

            papers.append({
                "doi": doi,
                "title": title,
                "authors": item.get("authors", ""),
                "abstract": abstract,
                "created_date": item.get("date"),
                "type": "preprint",
                "source": server,
            })

        cursor += len(collection)
        if len(collection) < 100:
            break

    return papers
