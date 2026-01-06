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

from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Set, Tuple

from langchain_core.documents import Document

from paperfind.config import ZOTERO_DB
from paperfind.db import (
    ZOTERO_SCHEMA,
    get_conn,
    is_postgres,
    placeholder,
    qualify_table,
    table_exists,
)
from paperfind.logging import get_logger
from paperfind.rerank import get_rerank_model, rerank_pairs
from paperfind.search.formatting import format_document, format_markdown_recommendation
from paperfind.search.utils import check_vector_store
from paperfind.types import (
    Recommendation,
    RecommendationList,
    RecommendationResult,
    RecommendationWithQuery,
    ZoteroPaper,
)
from paperfind.vectorstore import (
    get_embeddings_from_store,
    get_vector_store,
    similarity_search_by_vector,
    vector_store_exists,
)

logger = get_logger(__name__)


def _check_zotero_db() -> bool:
    """Check if Zotero database exists and has required tables."""
    if not is_postgres() and not Path(ZOTERO_DB).exists():
        logger.error("Zotero database not found. Run 'paperfind sync' first.")
        return False

    try:
        conn = get_conn(ZOTERO_SCHEMA)
    except Exception as exc:
        logger.error(f"Failed to connect to Zotero database: {exc}")
        return False

    has_items = table_exists(conn, ZOTERO_SCHEMA, "items")
    conn.close()

    if not has_items:
        logger.error("No Zotero items found. Run 'paperfind sync' first.")
        return False

    return True


def get_collection_id_by_name(collection_name: str) -> Optional[int]:
    """Look up collection id by collection name or key."""
    conn = get_conn(ZOTERO_SCHEMA)
    cur = conn.cursor()
    table = qualify_table(ZOTERO_SCHEMA, "collections")
    ph = placeholder()

    # Try by key first
    cur.execute(
        f"SELECT id FROM {table} WHERE collection_key = {ph}",
        (collection_name,)
    )
    row = cur.fetchone()

    if not row:
        # Try by name (case-insensitive)
        if is_postgres():
            cur.execute(
                f"SELECT id FROM {table} WHERE LOWER(name) = LOWER({ph})",
                (collection_name,)
            )
        else:
            cur.execute(
                f"SELECT id FROM {table} WHERE LOWER(name) = LOWER({ph})",
                (collection_name,)
            )
        row = cur.fetchone()

    conn.close()
    return row["id"] if row else None


def get_zotero_papers(collection: Optional[str] = None) -> List[ZoteroPaper]:
    """Get papers from Zotero database (includes zotero_key for embedding lookup)."""
    if not _check_zotero_db():
        return []

    conn = get_conn(ZOTERO_SCHEMA)
    cur = conn.cursor()
    items_table = qualify_table(ZOTERO_SCHEMA, "items")
    item_collections_table = qualify_table(ZOTERO_SCHEMA, "item_collections")
    ph = placeholder()

    if collection:
        collection_id = get_collection_id_by_name(collection)
        if collection_id:
            cur.execute(
                f"""
                SELECT i.zotero_key, i.title, i.abstract, i.doi
                FROM {items_table} i
                JOIN {item_collections_table} ic ON i.id = ic.item_id
                WHERE ic.collection_id = {ph}
                """,
                (collection_id,)
            )
        else:
            logger.warning(f"Collection '{collection}' not found.")
            conn.close()
            return []
    else:
        cur.execute(f"SELECT zotero_key, title, abstract, doi FROM {items_table}")

    papers = [dict(row) for row in cur.fetchall()]
    conn.close()
    return papers


def get_zotero_dois() -> Set[str]:
    """Get all DOIs from Zotero to exclude from recommendations."""
    if not is_postgres() and not Path(ZOTERO_DB).exists():
        return set()

    try:
        conn = get_conn(ZOTERO_SCHEMA)
    except Exception:
        return set()
    cur = conn.cursor()

    if not table_exists(conn, ZOTERO_SCHEMA, "items"):
        conn.close()
        return set()

    table = qualify_table(ZOTERO_SCHEMA, "items")
    cur.execute(f"SELECT doi FROM {table} WHERE doi IS NOT NULL")
    dois = {row["doi"].lower() for row in cur.fetchall() if row["doi"]}
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
    logger.info(f"Reranking {len(candidates)} candidates with {model_name}...")

    try:
        scores = rerank_pairs(pairs, model=model_name)
    except ImportError as exc:
        logger.warning(f"Rerank unavailable: {exc}")
        return _strip_query(recommendations[:k]), False
    except Exception as exc:
        logger.error(f"Rerank failed: {exc}")
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
    collection: Optional[str] = None,
    rerank: bool = False,
    rerank_candidates: int = 50,
    return_rerank_used: bool = False,
    max_age_days: Optional[int] = None,
    exclude_dois: Optional[Set[str]] = None,
) -> RecommendationResult:
    """
    Get paper recommendations based on Zotero library.

    Uses pre-embedded Zotero papers to find similar papers
    from the daily papers database.

    Args:
        k: Number of recommendations to return
        collection: Optional Zotero collection to base recommendations on
        rerank: Whether to use cross-encoder reranking
        rerank_candidates: Number of candidates to consider for reranking
        return_rerank_used: If True, returns (recommendations, rerank_used)
        max_age_days: Only recommend papers published within this many days
        exclude_dois: Set of DOIs to exclude from recommendations (e.g., previously sent)
    """
    def _return_empty() -> RecommendationResult:
        empty: RecommendationList = []
        return (empty, False) if return_rerank_used else empty

    # Check prerequisites
    if not check_vector_store():
        return _return_empty()

    # Check if Zotero embeddings exist
    if not vector_store_exists("zotero"):
        logger.error("Zotero embeddings not found. Run 'paperfind embed' first.")
        return _return_empty()

    # Get Zotero papers (includes zotero_key for embedding lookup)
    zotero_papers = get_zotero_papers(collection)
    if not zotero_papers:
        return _return_empty()

    # Get zotero_keys for embedding lookup
    zotero_keys = [p["zotero_key"] for p in zotero_papers if p.get("zotero_key")]
    if not zotero_keys:
        logger.error("No Zotero keys found in database.")
        return _return_empty()

    # Build lookup for paper metadata by zotero_key
    paper_by_key = {p["zotero_key"]: p for p in zotero_papers if p.get("zotero_key")}

    # Get DOIs to exclude (already in library)
    existing_dois = get_zotero_dois()

    # Load vector stores
    zotero_vectordb = get_vector_store("zotero")
    daily_vectordb = get_vector_store()

    # Get pre-computed embeddings from Zotero vector store
    logger.info(f"Loading embeddings for {len(zotero_keys)} Zotero papers...")
    embeddings = get_embeddings_from_store(zotero_vectordb, zotero_keys)

    if not embeddings:
        logger.error("No embeddings found. Run 'paperfind embed' first.")
        return _return_empty()

    missing_count = len(zotero_keys) - len(embeddings)
    if missing_count > 0:
        logger.warning(f"{missing_count} papers not embedded. Run 'paperfind embed' to embed all.")

    # Collect recommendations with scores
    recommendations = {}  # doi -> (score, doc, zotero_title, query_text)

    # Calculate cutoff date for filtering
    cutoff_date = None
    if max_age_days is not None:
        cutoff_date = date.today() - timedelta(days=max_age_days - 1)
        logger.info(f"Filtering to papers published since {cutoff_date}")

    logger.info(f"Finding papers similar to {len(embeddings)} embedded papers...")

    candidate_k = max(k, rerank_candidates) if rerank else k

    for zotero_key, embedding in embeddings.items():
        paper = paper_by_key.get(zotero_key, {})
        zotero_title = paper.get("title", "Unknown")

        # Build query text for reranking (title + abstract)
        query_parts = []
        if paper.get("title"):
            query_parts.append(paper["title"])
        if paper.get("abstract"):
            query_parts.append(paper["abstract"])
        query_text = " ".join(query_parts) if query_parts else zotero_title

        # Search daily papers using pre-computed embedding
        results = similarity_search_by_vector(daily_vectordb, embedding, k=candidate_k)

        for doc, score in results:
            doi = doc.metadata.get("doi", "")
            if not doi:
                continue

            # Skip if already in Zotero
            if doi.lower() in existing_dois:
                continue

            # Skip if in exclude list (e.g., previously sent)
            if exclude_dois and doi in exclude_dois:
                continue

            # Skip if paper is older than cutoff date
            if cutoff_date is not None:
                created_date_str = doc.metadata.get("created_date")
                if created_date_str:
                    try:
                        paper_date = date.fromisoformat(str(created_date_str)[:10])
                        if paper_date < cutoff_date:
                            continue
                    except (ValueError, TypeError):
                        pass  # Include papers with unparseable dates

            # Keep track of best score for each paper (with source Zotero paper)
            if doi not in recommendations or score < recommendations[doi][0]:
                recommendations[doi] = (score, doc, zotero_title, query_text)

    # Sort by score (lower is better for distance)
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1][0])

    if not rerank:
        results = _strip_query(sorted_recs[:k])
        return (results, False) if return_rerank_used else results

    reranked, rerank_used = _rerank_recommendations(sorted_recs, k, rerank_candidates)
    return (reranked, rerank_used) if return_rerank_used else reranked


def format_markdown(
    recommendations: RecommendationList,
    today: str,
    collection: Optional[str] = None,
    rerank: bool = False,
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
    collection: Optional[str] = None,
    output: Optional[str] = None,
    rerank: bool = False,
    rerank_candidates: int = 50,
    max_age_days: Optional[int] = None,
) -> None:
    """Run paper recommendations with parsed parameters."""
    logger.info("=" * 60)
    logger.info("Paper Recommendations Based on Your Zotero Library")
    logger.info("=" * 60)

    recommendations, rerank_used = get_recommendations(
        k=num_results,
        collection=collection,
        rerank=rerank,
        rerank_candidates=rerank_candidates,
        return_rerank_used=True,
        max_age_days=max_age_days,
    )

    if not recommendations:
        logger.warning("No recommendations found. Make sure you have:")
        logger.warning("  1. Synced your Zotero library (paperfind sync)")
        logger.warning("  2. Embedded your Zotero papers (paperfind embed)")
        logger.warning("  3. Fetched daily papers (paperfind fetch --rebuild-vectors)")
        return

    # Save to markdown file if output specified
    if output:
        today = date.today().isoformat()
        markdown = format_markdown(recommendations, today, collection, rerank=rerank_used)

        output_path = Path(output)
        output_path.write_text(markdown)
        logger.info(f"Saved {len(recommendations)} recommendations to {output}")
    else:
        # Print to console
        logger.info(f"Top {len(recommendations)} recommendations:")
        score_label = "Rerank score" if rerank_used else None
        show_similarity = not rerank_used
        for rank, (doi, (score, doc, zotero_title)) in enumerate(recommendations, 1):
            logger.info(format_document(
                doc,
                rank=rank,
                score=score,
                similar_to=zotero_title,
                show_score_as_similarity=show_similarity,
                score_label=score_label,
            ))
        logger.info("")
