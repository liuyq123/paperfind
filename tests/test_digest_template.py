from datetime import date

from langchain_core.documents import Document

from paperfind.digest.template import render_digest


def test_render_digest_uses_metadata_title_and_abstract() -> None:
    doc = Document(
        page_content="Fallback Title\n\nFallback abstract.",
        metadata={
            "title": "Meta Title",
            "abstract": "Meta abstract.",
            "authors": "A. Author",
            "source": "arxiv",
            "created_date": "2024-01-01",
        },
    )
    recommendations = [("arxiv:1234.5678", (0.1, doc, "Seed Paper"))]

    html = render_digest(recommendations, date(2024, 1, 2), rerank=False)

    assert "Meta Title" in html
    assert "Meta abstract." in html
    assert "https://arxiv.org/abs/1234.5678" in html


def test_render_digest_with_doi_link() -> None:
    doc = Document(
        page_content="Title\n\nAbstract.",
        metadata={
            "title": "DOI Paper",
            "abstract": "Abstract text.",
            "authors": "B. Author",
            "source": "crossref",
        },
    )
    recommendations = [("10.1234/test.5678", (0.2, doc, "Seed Paper"))]

    html = render_digest(recommendations, date(2024, 1, 2), rerank=False)

    assert "DOI Paper" in html
    assert "https://doi.org/10.1234/test.5678" in html


def test_render_digest_with_multiple_recommendations() -> None:
    docs = [
        Document(
            page_content=f"Title {i}\n\nAbstract {i}.",
            metadata={
                "title": f"Paper {i}",
                "abstract": f"Abstract {i}.",
                "authors": f"Author {i}",
                "source": "arxiv",
            },
        )
        for i in range(3)
    ]
    recommendations = [
        (f"arxiv:2401.0000{i}", (0.1 * i, docs[i], f"Seed {i}"))
        for i in range(3)
    ]

    html = render_digest(recommendations, date(2024, 1, 2), rerank=False)

    assert "Paper 0" in html
    assert "Paper 1" in html
    assert "Paper 2" in html


def test_render_digest_with_rerank_enabled() -> None:
    doc = Document(
        page_content="Title\n\nAbstract.",
        metadata={
            "title": "Reranked Paper",
            "abstract": "Abstract.",
            "authors": "C. Author",
            "source": "biorxiv",
        },
    )
    recommendations = [("10.1101/2024.01.01.000000", (0.9, doc, "Seed"))]

    html = render_digest(recommendations, date(2024, 1, 2), rerank=True)

    assert "Reranked Paper" in html
    # Rerank score should be displayed differently
    assert "0.9" in html or "rerank" in html.lower()


def test_render_digest_empty_recommendations() -> None:
    html = render_digest([], date(2024, 1, 2), rerank=False)

    # Should still render valid HTML
    assert "<html" in html.lower() or "<!doctype" in html.lower() or "paper" in html.lower()
