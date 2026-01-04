import importlib
import pytest
from langchain_core.documents import Document

pytest.importorskip("fastapi")

from paperfind.api import recommend, search_papers
import paperfind.search.utils as utils_module

# Import modules directly to avoid name shadowing from __init__.py
search_module = importlib.import_module("paperfind.search.search")
recommend_module = importlib.import_module("paperfind.search.recommend")


def test_search_papers_uses_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = Document(
        page_content="Fallback Title\n\nFallback abstract.",
        metadata={
            "title": "Meta Title",
            "abstract": "Meta Abstract",
            "authors": "A. Author",
            "source": "crossref",
        },
    )

    monkeypatch.setattr(utils_module, "check_vector_store", lambda *args, **kwargs: True)
    monkeypatch.setattr(search_module, "search", lambda *args, **kwargs: [doc])

    response = search_papers(
        query="test query",
        k=5,
        source="daily_papers",
        scores=False,
        rag=False,
    )

    assert response.count == 1
    assert response.results[0].title == "Meta Title"
    assert response.results[0].abstract == "Meta Abstract"
    assert response.results[0].authors == "A. Author"
    assert response.results[0].source == "crossref"


def test_search_papers_with_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = Document(
        page_content="Fallback Title\n\nFallback abstract.",
        metadata={"title": "Meta Title", "abstract": "Meta Abstract"},
    )

    monkeypatch.setattr(utils_module, "check_vector_store", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        search_module,
        "search_with_scores",
        lambda *args, **kwargs: [(doc, 0.42)],
    )

    response = search_papers(
        query="test query",
        k=5,
        source="daily_papers",
        scores=True,
        rag=False,
    )

    assert response.count == 1
    assert response.results[0].title == "Meta Title"
    assert response.results[0].abstract == "Meta Abstract"
    assert response.results[0].score == 0.42


def test_recommend_uses_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = Document(
        page_content="Fallback Title\n\nFallback abstract.",
        metadata={
            "title": "Meta Title",
            "abstract": "Meta Abstract",
            "authors": "A. Author",
            "source": "crossref",
            "created_date": "2024-01-01",
        },
    )
    recommendations = [("10.1234/example", (0.12, doc, "Seed Paper"))]

    monkeypatch.setattr(utils_module, "check_vector_store", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        recommend_module,
        "get_recommendations",
        lambda *args, **kwargs: (recommendations, True),
    )

    response = recommend(
        k=10,
        collection=None,
        rerank=True,
        rerank_candidates=50,
    )

    assert response.count == 1
    assert response.reranked is True
    assert response.recommendations[0].title == "Meta Title"
    assert response.recommendations[0].abstract == "Meta Abstract"
    assert response.recommendations[0].authors == "A. Author"
    assert response.recommendations[0].created_date == "2024-01-01"
    assert response.recommendations[0].similar_to == "Seed Paper"
