"""
recommend.py

Recommend papers based on your Zotero library.

Usage:
    paperfind recommend                              # Get today's recommendations
    paperfind recommend -k 20                        # Get top 20 recommendations
    paperfind recommend --collection "active learning"  # Recommend based on specific collection
    paperfind recommend -o recommendations.md        # Save to markdown file
"""

import sqlite3
from datetime import date
from pathlib import Path
from typing import List, Dict, Set

from langchain_chroma import Chroma

from paperfind.config import get_chroma_store_dir, ZOTERO_DB
from paperfind.embeddings import get_embeddings
from paperfind.search.formatting import format_document, format_markdown_recommendation


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


def get_recommendations(
    k: int = 10,
    collection: str = None,
) -> List[Dict]:
    """
    Get paper recommendations based on Zotero library.

    Uses each Zotero paper as a query and finds similar papers
    from the daily papers database.
    """
    # Get Zotero papers to use as queries
    zotero_papers = get_zotero_papers(collection)
    if not zotero_papers:
        print("No papers found in Zotero library.")
        return []

    # Get DOIs to exclude (already in library)
    existing_dois = get_zotero_dois()

    # Load the daily papers vector store
    embeddings = get_embeddings()
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=get_chroma_store_dir(),
    )

    # Collect recommendations with scores
    recommendations = {}  # doi -> (score, doc, zotero_title)

    print(f"Finding papers similar to {len(zotero_papers)} papers in your library...")

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
        results = vectordb.similarity_search_with_score(query, k=k)

        for doc, score in results:
            doi = doc.metadata.get("doi", "")
            if not doi:
                continue

            # Skip if already in Zotero
            if doi.lower() in existing_dois:
                continue

            # Keep track of best score for each paper (with source Zotero paper)
            if doi not in recommendations or score < recommendations[doi][0]:
                recommendations[doi] = (score, doc, zotero_title)

    # Sort by score (lower is better for distance)
    sorted_recs = sorted(recommendations.items(), key=lambda x: x[1][0])

    return sorted_recs[:k]


def format_markdown(recommendations: List, today: str, collection: str = None) -> str:
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

    for rank, (doi, (score, doc, zotero_title)) in enumerate(recommendations, 1):
        lines.append(format_markdown_recommendation(rank, doi, score, doc, zotero_title))

    return "\n".join(lines)


def run_recommend(
    num_results: int = 10,
    collection: str = None,
    output: str = None,
) -> None:
    """Run paper recommendations with parsed parameters."""
    print(f"\n{'='*60}")
    print("Paper Recommendations Based on Your Zotero Library")
    print(f"{'='*60}")

    recommendations = get_recommendations(
        k=num_results,
        collection=collection,
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
        markdown = format_markdown(recommendations, today, collection)

        output_path = Path(output)
        output_path.write_text(markdown)
        print(f"\nSaved {len(recommendations)} recommendations to {output}")
    else:
        # Print to console
        print(f"\nTop {len(recommendations)} recommendations:\n")
        for rank, (doi, (score, doc, zotero_title)) in enumerate(recommendations, 1):
            print(format_document(
                doc,
                rank=rank,
                score=score,
                similar_to=zotero_title,
                show_score_as_similarity=True,
            ))
        print()
