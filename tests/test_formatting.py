from langchain_core.documents import Document

from paperfind.search.formatting import format_document, format_markdown_recommendation


def test_format_document_uses_metadata_fields() -> None:
    doc = Document(
        page_content="Fallback Title\n\nFallback abstract.",
        metadata={
            "title": "Meta Title",
            "abstract": "Meta abstract.",
            "authors": "A. Author",
            "doi": "10.1234/example",
            "source": "crossref",
            "created_date": "2024-01-01",
        },
    )

    output = format_document(doc, rank=1)

    assert "Title: Meta Title" in output
    assert "Authors: A. Author" in output
    assert "DOI: 10.1234/example" in output
    assert "Source: crossref" in output
    assert "Date: 2024-01-01" in output
    assert "Abstract: Meta abstract." in output


def test_format_markdown_recommendation_uses_metadata_fields() -> None:
    doc = Document(
        page_content="Fallback Title\n\nFallback abstract.",
        metadata={
            "title": "Meta Title",
            "abstract": "Meta abstract.",
            "authors": "A. Author",
            "source": "crossref",
            "created_date": "2024-01-01",
        },
    )

    output = format_markdown_recommendation(
        rank=1,
        doi="10.1234/example",
        score=0.1,
        doc=doc,
        similar_to="Related Paper",
        show_score_as_similarity=False,
    )

    assert "## 1. Meta Title" in output
    assert "**Authors:** A. Author" in output
    assert "**Date:** 2024-01-01 | **Source:** crossref" in output
    assert "**Link:** [10.1234/example](https://doi.org/10.1234/example)" in output
    assert "**Abstract:**" in output


def test_format_document_with_score() -> None:
    doc = Document(
        page_content="Title\n\nAbstract text.",
        metadata={"title": "Title"},
    )

    output = format_document(doc, rank=1, score=0.5)

    assert "#1" in output
    assert "score: 0.5000" in output


def test_format_document_with_similarity_score() -> None:
    doc = Document(
        page_content="Title\n\nAbstract text.",
        metadata={"title": "Title"},
    )

    output = format_document(doc, rank=1, score=0.5, show_score_as_similarity=True)

    assert "similarity:" in output


def test_format_document_with_score_label() -> None:
    doc = Document(
        page_content="Title\n\nAbstract text.",
        metadata={"title": "Title"},
    )

    output = format_document(doc, rank=1, score=0.5, score_label="Rerank")

    assert "Rerank: 0.5000" in output


def test_format_document_truncates_long_title() -> None:
    long_title = "A" * 150
    doc = Document(
        page_content=f"{long_title}\n\nAbstract.",
        metadata={},
    )

    output = format_document(doc, rank=1)

    assert "A" * 100 + "..." in output
    assert "A" * 101 not in output


def test_format_document_truncates_long_authors() -> None:
    long_authors = "Author Name, " * 20
    doc = Document(
        page_content="Title\n\nAbstract.",
        metadata={"authors": long_authors},
    )

    output = format_document(doc, rank=1)

    assert "..." in output


def test_format_markdown_recommendation_with_arxiv_doi() -> None:
    doc = Document(
        page_content="Title\n\nAbstract.",
        metadata={"title": "Title", "source": "arxiv"},
    )

    output = format_markdown_recommendation(
        rank=1,
        doi="arxiv:2401.12345",
        score=0.1,
        doc=doc,
        similar_to="Related",
    )

    assert "https://arxiv.org/abs/2401.12345" in output


def test_format_markdown_recommendation_similarity_display() -> None:
    doc = Document(
        page_content="Title\n\nAbstract.",
        metadata={"title": "Title", "source": "arxiv"},
    )

    output = format_markdown_recommendation(
        rank=1,
        doi="10.1234/test",
        score=0.0,  # score=0 gives 100% similarity
        doc=doc,
        similar_to="Related Paper",
        show_score_as_similarity=True,
    )

    assert "**Similarity:**" in output
    assert "100" in output  # 100% similarity
