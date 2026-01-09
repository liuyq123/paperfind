"""CrossRef fetcher."""


import html
import re
from datetime import date
from typing import Any, Dict, List, Optional

import requests

from paperfind.config import CROSSREF_EMAIL
from paperfind.logging import get_logger
from paperfind.retry import retry_request
from paperfind.types import PaperDict

logger = get_logger(__name__)

CROSSREF_API = "https://api.crossref.org/works"
TOOL_NAME = "paperfind"
TOOL_VERSION = "0.1"


def fetch_crossref(target_date: date, type_filter: Optional[str] = None) -> List[PaperDict]:
    """Fetch papers from CrossRef for a specific date."""
    papers: List[PaperDict] = []

    if CROSSREF_EMAIL:
        user_agent = f"{TOOL_NAME}/{TOOL_VERSION} (mailto:{CROSSREF_EMAIL})"
    else:
        user_agent = f"{TOOL_NAME}/{TOOL_VERSION}"

    params: Dict[str, Any] = {
        "filter": f"from-created-date:{target_date},until-created-date:{target_date}",
        "rows": 1000,
        "cursor": "*",
        "mailto": CROSSREF_EMAIL,
    }
    if type_filter:
        params["filter"] += f",type:{type_filter}"

    headers = {"User-Agent": user_agent}

    while True:
        try:
            resp = retry_request(
                lambda: requests.get(CROSSREF_API, params=params, headers=headers, timeout=60),
                description="CrossRef",
            )
            resp.raise_for_status()
            msg = resp.json()["message"]
        except requests.RequestException as e:
            logger.error(f"Error fetching CrossRef: {e}")
            break

        items = msg.get("items", [])
        if not items:
            break

        for item in items:
            title = (item.get("title") or [""])[0]

            authors_list = item.get("author", []) or []
            authors = []
            for a in authors_list:
                given = a.get("given", "") or ""
                family = a.get("family", "") or ""
                name = f"{given} {family}".strip()
                if name:
                    authors.append(name)
            authors_str = ", ".join(authors)

            created = item.get("created", {}) or {}
            c_parts = (created.get("date-parts") or [[]])[0]
            created_date = None
            if c_parts:
                year = f"{c_parts[0]:04d}"
                month = f"{c_parts[1]:02d}" if len(c_parts) > 1 else "01"
                day = f"{c_parts[2]:02d}" if len(c_parts) > 2 else "01"
                created_date = f"{year}-{month}-{day}"

            raw_abstract = item.get("abstract")
            if raw_abstract:
                no_tags = re.sub(r"<[^>]+>", " ", raw_abstract)
                abstract = html.unescape(no_tags).strip()
            else:
                abstract = None

            if not item.get("DOI") or not title or not abstract:
                continue

            papers.append({
                "doi": item.get("DOI"),
                "title": title,
                "authors": authors_str,
                "abstract": abstract,
                "created_date": created_date,
                "type": item.get("type", "article"),
                "source": "crossref",
            })

        next_cursor = msg.get("next-cursor")
        if not next_cursor:
            break
        params["cursor"] = next_cursor

    return papers
