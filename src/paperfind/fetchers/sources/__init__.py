"""Source-specific fetchers."""

from paperfind.config import ARXIV_CATEGORIES, BIORXIV_CATEGORIES

from .arxiv import fetch_arxiv
from .biorxiv import fetch_biorxiv
from .chemrxiv import fetch_chemrxiv
from .crossref import fetch_crossref

__all__ = [
    "ARXIV_CATEGORIES",
    "BIORXIV_CATEGORIES",
    "fetch_arxiv",
    "fetch_biorxiv",
    "fetch_chemrxiv",
    "fetch_crossref",
]
