"""arXiv fetcher."""

from datetime import date, timedelta
from typing import Dict, List
from urllib.parse import urlencode
import xml.etree.ElementTree as ET

import requests

from paperfind.config import ARXIV_CATEGORIES

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def fetch_arxiv(category: str, days: int = 7, max_results: int = 100) -> List[Dict]:
    """Fetch papers from arXiv for a category."""
    papers = []
    query = f"cat:{category}"

    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    url = f"{ARXIV_API}?{urlencode(params)}"

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError) as e:
        print(f"    Error fetching arXiv {category}: {e}")
        return papers

    cutoff_date = date.today() - timedelta(days=days)

    for entry in root.findall("atom:entry", ARXIV_NAMESPACES):
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
        abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""

        published_elem = entry.find("atom:published", ARXIV_NAMESPACES)
        created_date = published_elem.text[:10] if published_elem is not None else None

        if created_date:
            paper_date = date.fromisoformat(created_date)
            if paper_date < cutoff_date:
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

    return papers
