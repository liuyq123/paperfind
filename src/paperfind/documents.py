"""Helpers for extracting structured fields from LangChain documents."""

from __future__ import annotations

from typing import Optional, Tuple

from langchain_core.documents import Document


def extract_title_and_abstract(doc: Document) -> Tuple[str, Optional[str]]:
    """Return (title, abstract) using metadata first, with content fallback."""
    metadata = doc.metadata or {}
    title = metadata.get("title")
    abstract = metadata.get("abstract")

    content = doc.page_content or ""

    if not title:
        title = _extract_title_from_content(content)
    if abstract is None:
        abstract = _extract_abstract_from_content(content)

    if title is None:
        title = ""

    if abstract is not None:
        abstract = abstract.strip()
        if not abstract:
            abstract = None

    return title, abstract


def _extract_title_from_content(content: str) -> str:
    if not content:
        return ""
    if "\n\n" in content:
        return content.split("\n\n", 1)[0].strip()
    return content.split("\n", 1)[0].strip()


def _extract_abstract_from_content(content: str) -> Optional[str]:
    if not content:
        return None

    abstract: Optional[str] = None
    if "\n\n" in content:
        abstract = content.split("\n\n", 1)[1]
    else:
        lines = content.split("\n")
        if len(lines) > 1:
            abstract = "\n".join(lines[1:])

    if abstract is None:
        return None

    abstract = abstract.strip()

    tags_marker = "\n\nTags:"
    if tags_marker in abstract:
        abstract = abstract.split(tags_marker, 1)[0].strip()

    if abstract.lower().startswith("abstract:"):
        abstract = abstract[len("abstract:"):].strip()

    return abstract if abstract else None
