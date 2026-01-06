"""Shared type definitions for Paperfind."""

from typing import Any, Dict, List, Tuple, Union

from langchain_core.documents import Document

# Paper metadata as fetched from sources or Zotero
PaperDict = Dict[str, Any]

# Zotero item row for database storage
ZoteroItemRow = Dict[str, Any]

# Zotero paper dict used in recommendations
ZoteroPaper = Dict[str, Any]

# Recommendation tuple: (doi, (score, document, zotero_title))
# - doi: DOI of the recommended paper
# - score: distance from vector search (lower is better) or rerank score (higher is better)
# - document: langchain Document with paper content and metadata
# - zotero_title: title of the Zotero paper this recommendation is based on
Recommendation = Tuple[str, Tuple[float, Document, str]]

# Recommendation with query text: (doi, (score, document, zotero_title, query_text))
# Extended recommendation that includes the query text used for similarity search
RecommendationWithQuery = Tuple[str, Tuple[float, Document, str, str]]

# List of recommendations
RecommendationList = List[Recommendation]

# Return type for get_recommendations (optionally includes rerank flag)
RecommendationResult = Union[RecommendationList, Tuple[RecommendationList, bool]]

# Search result with score: (document, score)
SearchResultWithScore = Tuple[Document, float]
