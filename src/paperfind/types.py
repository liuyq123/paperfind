"""Shared type definitions for Paperfind."""

from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

# Paper metadata as fetched from sources or Zotero
PaperDict = Dict[str, Any]

# Zotero item row for database storage
ZoteroItemRow = Dict[str, Any]

# Recommendation tuple: (doi, (score, document, zotero_title))
Recommendation = Tuple[str, Tuple[float, Document, str]]

# Recommendation with query text: (doi, (score, document, zotero_title, query_text))
RecommendationWithQuery = Tuple[str, Tuple[float, Document, str, str]]

# List of recommendations
RecommendationList = List[Recommendation]

# Search result with score: (document, score)
SearchResultWithScore = Tuple[Document, float]
