"""Fetchers module for paperfind.

This module provides functions to fetch papers from various sources:
- CrossRef (journal articles and preprints)
- bioRxiv/medRxiv (preprints)
- arXiv (preprints)
"""

from .db import init_db, upsert_work
from .fetch_papers import fetch_all
from .sources import (
    ARXIV_CATEGORIES,
    BIORXIV_CATEGORIES,
    fetch_arxiv,
    fetch_biorxiv,
    fetch_crossref,
)
from .vector import rebuild_vectors

__all__ = [
    "init_db",
    "upsert_work",
    "fetch_crossref",
    "fetch_biorxiv",
    "fetch_arxiv",
    "fetch_all",
    "rebuild_vectors",
    "ARXIV_CATEGORIES",
    "BIORXIV_CATEGORIES",
]
