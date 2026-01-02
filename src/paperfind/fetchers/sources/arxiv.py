"""arXiv fetcher."""

import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import List
from urllib.parse import urlencode

import requests

from paperfind.config import ARXIV_CATEGORIES
from paperfind.logging import get_logger
from paperfind.types import PaperDict

logger = get_logger(__name__)

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


PAGE_SIZE = 100
MAX_PAGES = 10  # Safety limit: 1000 papers max per category
REQUEST_DELAY_SECONDS = 3


def fetch_arxiv(category: str, days: int = 7) -> List[PaperDict]:
    """Fetch papers from arXiv for a category with pagination."""
    papers: List[PaperDict] = []
    query = f"cat:{category}"
    cutoff_date = date.today() - timedelta(days=days)
    start = 0

    for _ in range(MAX_PAGES):
        params = {
            "search_query": query,
            "start": start,
            "max_results": PAGE_SIZE,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        url = f"{ARXIV_API}?{urlencode(params)}"

        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except (requests.RequestException, ET.ParseError) as e:
            logger.error(f"Error fetching arXiv {category}: {e}")
            break

        entries = root.findall("atom:entry", ARXIV_NAMESPACES)
        if not entries:
            break

        reached_cutoff = False

        for entry in entries:
            id_elem = entry.find("atom:id", ARXIV_NAMESPACES)
            if id_elem is None:
                continue

            arxiv_url = id_elem.text
            arxiv_id = arxiv_url.split("/abs/")[-1] if arxiv_url else None
            if not arxiv_id:
                continue

            doi = f"arxiv:{arxiv_id.split('v')[0]}"

            title_elem = entry.find("atom:title", ARXIV_NAMESPACES)
            title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else ""

            authors = []
            for author in entry.findall("atom:author", ARXIV_NAMESPACES):
                name_elem = author.find("atom:name", ARXIV_NAMESPACES)
                if name_elem is not None:
                    authors.append(name_elem.text)
            authors_str = ", ".join(authors)

            summary_elem = entry.find("atom:summary", ARXIV_NAMESPACES)
            abstract = (
                summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""
            )

            published_elem = entry.find("atom:published", ARXIV_NAMESPACES)
            created_date = published_elem.text[:10] if published_elem is not None else None

            if created_date:
                paper_date = date.fromisoformat(created_date)
                if paper_date < cutoff_date:
                    reached_cutoff = True
                    continue

            if not title or not abstract:
                continue

            papers.append({
                "doi": doi,
                "title": title,
                "authors": authors_str,
                "abstract": abstract,
                "created_date": created_date,
                "type": "preprint",
                "source": f"arxiv:{category}",
            })

        # Stop if we got fewer results than requested (end of results)
        # or if we've reached papers older than cutoff
        if len(entries) < PAGE_SIZE or reached_cutoff:
            break

        time.sleep(REQUEST_DELAY_SECONDS)
        start += PAGE_SIZE

    return papers
