"""Shared formatting utilities for search results and recommendations."""

from typing import Optional

from langchain_core.documents import Document


def format_document(
    doc: Document,
    rank: int,
    score: Optional[float] = None,
    similar_to: Optional[str] = None,
    show_score_as_similarity: bool = False,
) -> str:
    """
    Format a document for console display.

    Args:
        doc: The document to format
        rank: Display rank (1-indexed)
        score: Optional similarity/distance score
        similar_to: Optional title of the paper this is similar to
        show_score_as_similarity: If True, convert score to similarity percentage

    Returns:
        Formatted string for console output
    """
    lines = [f"\n{'='*60}"]

    # Header line with rank and score
    header = f"#{rank}"
    if score is not None:
        if show_score_as_similarity:
            similarity = 1 / (1 + score)
            header += f" (similarity: {similarity:.2%})"
        else:
            header += f" (score: {score:.4f})"
    lines.append(header)

    # Similar to line (for recommendations)
    if similar_to:
        truncated = similar_to[:60] + "..." if len(similar_to) > 60 else similar_to
        lines.append(f"Similar to: {truncated}")

    lines.append("=" * 60)

    # Extract title and abstract from content
    content_lines = doc.page_content.split("\n")
    title = content_lines[0]
    if len(title) > 100:
        title = title[:100] + "..."
    lines.append(f"Title: {title}")

    # Metadata
    metadata = doc.metadata
    if metadata.get("doi"):
        lines.append(f"DOI: {metadata['doi']}")
    if metadata.get("authors"):
        authors = metadata["authors"]
        if len(authors) > 80:
            authors = authors[:80] + "..."
        lines.append(f"Authors: {authors}")
    if metadata.get("created_date"):
        lines.append(f"Date: {metadata['created_date']}")
    if metadata.get("source"):
        lines.append(f"Source: {metadata['source']}")

    # Abstract preview
    if len(content_lines) > 1:
        abstract = " ".join(content_lines[1:])[:300]
        if len(abstract) == 300:
            abstract += "..."
        lines.append(f"\nAbstract: {abstract}")

    return "\n".join(lines)


def format_markdown_recommendation(
    rank: int,
    doi: str,
    score: float,
    doc: Document,
    similar_to: str,
) -> str:
    """
    Format a single recommendation as markdown.

    Args:
        rank: Display rank (1-indexed)
        doi: Paper DOI
        score: Distance score
        doc: The document
        similar_to: Title of the Zotero paper this is similar to

    Returns:
        Markdown formatted string
    """
    content_lines = doc.page_content.split("\n")
    title = content_lines[0]
    authors = doc.metadata.get("authors", "Unknown")
    source = doc.metadata.get("source", "")
    pub_date = doc.metadata.get("created_date", "")

    # Get full abstract
    abstract = ""
    if len(content_lines) > 1:
        abstract = " ".join(content_lines[1:]).strip()

    # Build DOI link
    doi_link = ""
    if doi:
        if doi.startswith("arxiv:"):
            arxiv_id = doi.replace("arxiv:", "")
            doi_link = f"https://arxiv.org/abs/{arxiv_id}"
        else:
            doi_link = f"https://doi.org/{doi}"

    similarity = 1 / (1 + score)

    lines = [
        f"## {rank}. {title}",
        f"",
        f"**Similarity:** {similarity:.1%} to *{similar_to}*",
        f"",
        f"**Authors:** {authors}",
        f"",
    ]

    if pub_date:
        lines.append(f"**Date:** {pub_date} | **Source:** {source}")
    else:
        lines.append(f"**Source:** {source}")
    lines.append(f"")

    if doi_link:
        lines.append(f"**Link:** [{doi}]({doi_link})")
        lines.append(f"")

    if abstract:
        lines.append(f"**Abstract:**")
        lines.append(f"")
        lines.append(f"> {abstract}")
        lines.append(f"")

    lines.append(f"---")
    lines.append(f"")

    return "\n".join(lines)
