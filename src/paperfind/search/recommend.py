"""
recommend.py

Recommend papers based on your Zotero library.

Usage:
    paperfind recommend                              # Get today's recommendations
    paperfind recommend -k 20                        # Get top 20 recommendations
    paperfind recommend --collection "active learning"  # Recommend based on specific collection
    paperfind recommend -o recommendations.md        # Save to markdown file
    paperfind recommend --no-rerank                  # Disable reranking
"""

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, List, Dict, Set, Tuple, Union

from langchain_chroma import Chroma

from paperfind.config import get_chroma_store_dir, ZOTERO_DB
from paperfind.embeddings import get_embeddings
from paperfind.rerank import get_rerank_model, rerank_pairs
from paperfind.search.formatting import format_document, format_markdown_recommendation
from paperfind.search.utils import check_vector_store

Recommendation = Tuple[str, Tuple[float, Any, str]]
RecommendationWithQuery = Tuple[str, Tuple[float, Any, str, str]]
RecommendationList = List[Recommendation]
RecommendationResult = Union[RecommendationList, Tuple[RecommendationList, bool]]


def _check_zotero_db() -> bool:
    """Check if Zotero database exists and has required tables."""
    if not Path(ZOTERO_DB).exists():
        print("Error: Zotero database not found. Run 'paperfind sync' first.")
        return False

    conn = sqlite3.connect(ZOTERO_DB)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='items'")
    has_items = cur.fetchone() is not None
    conn.close()

    if not has_items:
        print("Error: No Zotero items found. Run 'paperfind sync' first.")
        return False

    return True


def get_project_id_by_name(collection_name: str) -> int:
    """Look up project_id by collection name."""
    conn = sqlite3.connect(ZOTERO_DB)
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM projects WHERE name = ? OR collection_name = ?",
        (collection_name, collection_name)
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def get_zotero_papers(collection: str = None) -> List[Dict]:
    """Get papers from Zotero database."""
    if not _check_zotero_db():
        return []

    conn = sqlite3.connect(ZOTERO_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if collection:
        project_id = get_project_id_by_name(collection)
        if project_id:
            cur.execute(
                "SELECT title, abstract, doi FROM items WHERE project_id = ?",
                (project_id,)
            )
        else:
            print(f"Collection '{collection}' not found.")
            conn.close()
            return []
    else:
        cur.execute("SELECT title, abstract, doi FROM items")

    papers = [dict(row) for row in cur.fetchall()]
    conn.close()
    return papers


def get_zotero_dois() -> Set[str]:
    """Get all DOIs from Zotero to exclude from recommendations."""
    if not Path(ZOTERO_DB).exists():
        return set()

    conn = sqlite3.connect(ZOTERO_DB)
    cur = conn.cursor()

    # Check if items table exists
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='items'")
    if not cur.fetchone():
        conn.close()
        return set()

    cur.execute("SELECT doi FROM items WHERE doi IS NOT NULL")
    dois = {row[0].lower() for row in cur.fetchall() if row[0]}
    conn.close()
    return dois


def _strip_query(
    recommendations: List[RecommendationWithQuery],
) -> RecommendationList:
    return [
        (doi, (score, doc, zotero_title))
        for doi, (score, doc, zotero_title, _) in recommendations
    ]


def _rerank_recommendations(
    recommendations: List[RecommendationWithQuery],
    k: int,
    rerank_candidates: int,
) -> Tuple[RecommendationList, bool]:
    candidate_count = max(k, rerank_candidates)
    candidates = recommendations[:candidate_count]
    if not candidates:
        return [], False

    pairs = [
        (query_text, doc.page_content)
        for _, (_, doc, _, query_text) in candidates
    ]

    model_name = get_rerank_model()
    print(f"Reranking {len(candidates)} candidates with {model_name}...")

    try:
        scores = rerank_pairs(pairs, model=model_name)
    except ImportError as exc:
        print(f"Rerank unavailable: {exc}")
        return _strip_query(recommendations[:k]), False
    except Exception as exc:
        print(f"Rerank failed: {exc}")
        return _strip_query(recommendations[:k]), False

    ranked = sorted(
        zip(candidates, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    reranked = [
        (doi, (rerank_score, doc, zotero_title))
        for (doi, (_, doc, zotero_title, _)), rerank_score in ranked
    ]

    return reranked[:k], True


def get_recommendations(
    k: int = 10,
    collection: str = None,
    rerank: bool = True,
    rerank_candidates: int = 50,
    return_rerank_used: bool = False,
) -> RecommendationResult:
    """
    Get paper recommendations based on Zotero library.

    Uses each Zotero paper as a query and finds similar papers
    from the daily papers database.

    If return_rerank_used is True, returns (recommendations, rerank_used).
    """
    def _return_empty() -> RecommendationResult:
        empty: RecommendationList = []
        return (empty, False) if return_rerank_used else empty

    # Check prerequisites
    if not check_vector_store():
        return _return_empty()

    # Get Zotero papers to use as queries
    zotero_papers = get_zotero_papers(collection)
    if not zotero_papers:
        return _return_empty()

    # Get DOIs to exclude (already in library)
    existing_dois = get_zotero_dois()

    # Load the daily papers vector store
    embeddings = get_embeddings()
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=get_chroma_store_dir(),
    )

    # Collect recommendations with scores
    recommendations = {}  # doi -> (score, doc, zotero_title, query_text)

    print(f"Finding papers similar to {len(zotero_papers)} papers in your library...")

    candidate_k = max(k, rerank_candidates) if rerank else k

    for paper in zotero_papers:
        # Build query from title + abstract
        query_parts = []
        if paper.get("title"):
            query_parts.append(paper["title"])
        if paper.get("abstract"):
            query_parts.append(paper["abstract"])

        if not query_parts:
            continue

        query = " ".join(query_parts)
        zotero_title = paper.get("title", "Unknown")

        # Search for similar papers
        results = vectordb.similarity_search_with_score(query, k=candidate_k)

        for doc, score in results:
            doi = doc.metadata.get("doi", "")
            if not doi:
                continue

            # Skip if already in Zotero
            if doi.lower() in existing_dois:
                continue

            # Keep track of best score for each paper (with source Zotero paper)
            if doi not in recommendations or score < recommendations[doi][0]:
                recommendations[doi] = (score, doc, zotero_title, query)

    # Sort by score (lower is better for distance)
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1][0])

    if not rerank:
        results = _strip_query(sorted_recs[:k])
        return (results, False) if return_rerank_used else results

    reranked, rerank_used = _rerank_recommendations(sorted_recs, k, rerank_candidates)
    return (reranked, rerank_used) if return_rerank_used else reranked


def format_markdown(
    recommendations: List,
    today: str,
    collection: str = None,
    rerank: bool = True,
) -> str:
    """Format recommendations as a markdown document."""
    lines = [
        f"# Paper Recommendations",
        f"",
    ]
    if collection:
        lines.append(f"*Based on collection: {collection}*")
        lines.append(f"")
    lines.append(f"*Generated on {today}*")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    score_label = "Rerank score" if rerank else None
    show_similarity = not rerank

    for rank, (doi, (score, doc, zotero_title)) in enumerate(recommendations, 1):
        lines.append(
            format_markdown_recommendation(
                rank,
                doi,
                score,
                doc,
                zotero_title,
                score_label=score_label,
                show_score_as_similarity=show_similarity,
            )
        )

    return "\n".join(lines)


def run_recommend(
    num_results: int = 10,
    collection: str = None,
    output: str = None,
    rerank: bool = True,
    rerank_candidates: int = 50,
) -> None:
    """Run paper recommendations with parsed parameters."""
    print(f"\n{'='*60}")
    print("Paper Recommendations Based on Your Zotero Library")
    print(f"{'='*60}")

    recommendations, rerank_used = get_recommendations(
        k=num_results,
        collection=collection,
        rerank=rerank,
        rerank_candidates=rerank_candidates,
        return_rerank_used=True,
    )

    if not recommendations:
        print("\nNo recommendations found. Make sure you have:")
        print("  1. Papers in your Zotero library (paperfind sync)")
        print("  2. Fetched daily papers (paperfind fetch)")
        print("  3. Built vector embeddings (paperfind fetch --rebuild-vectors)")
        return

    # Save to markdown file if output specified
    if output:
        today = date.today().isoformat()
        markdown = format_markdown(recommendations, today, collection, rerank=rerank_used)

        output_path = Path(output)
        output_path.write_text(markdown)
        print(f"\nSaved {len(recommendations)} recommendations to {output}")
    else:
        # Print to console
        print(f"\nTop {len(recommendations)} recommendations:\n")
        score_label = "Rerank score" if rerank_used else None
        show_similarity = not rerank_used
        for rank, (doi, (score, doc, zotero_title)) in enumerate(recommendations, 1):
            print(format_document(
                doc,
                rank=rank,
                score=score,
                similar_to=zotero_title,
                show_score_as_similarity=show_similarity,
                score_label=score_label,
            ))
        print()
