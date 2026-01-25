import importlib
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

pytest.importorskip("fastapi")

from fastapi import BackgroundTasks

from paperfind.api import fetch, list_papers, recommend, search_papers
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
    def fake_get_recommendations(*args, **kwargs):
        assert "rerank" in kwargs
        assert "llm_rerank" not in kwargs
        assert kwargs["rerank"] is True
        assert kwargs.get("return_rerank_used") is True
        return recommendations, True

    monkeypatch.setattr(recommend_module, "get_recommendations", fake_get_recommendations)

    response = recommend(
        k=10,
        collection=None,
        rerank=True,
    )

    assert response.count == 1
    assert response.reranked is True
    assert response.recommendations[0].title == "Meta Title"
    assert response.recommendations[0].abstract == "Meta Abstract"
    assert response.recommendations[0].authors == "A. Author"
    assert response.recommendations[0].created_date == "2024-01-01"
    assert response.recommendations[0].similar_to == "Seed Paper"


class TestListPapersSourceFilter:
    """Tests for /papers endpoint source filtering."""

    @patch("paperfind.api.get_conn")
    def test_arxiv_source_uses_like_prefix(self, mock_get_conn: MagicMock) -> None:
        """Filtering by 'arxiv' should use LIKE to match arxiv:* sources."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # Mock count query
        mock_cursor.fetchone.side_effect = [
            {"total": 2},  # count result
        ]
        # Mock select query
        mock_cursor.fetchall.return_value = [
            {
                "doi": "arxiv:2401.12345",
                "title": "Paper 1",
                "authors": "Author 1",
                "abstract": "Abstract 1",
                "source": "arxiv:cs.AI",
                "created_date": "2024-01-15",
            },
            {
                "doi": "arxiv:2401.67890",
                "title": "Paper 2",
                "authors": "Author 2",
                "abstract": "Abstract 2",
                "source": "arxiv:q-bio.NC",
                "created_date": "2024-01-14",
            },
        ]
        mock_get_conn.return_value = mock_conn

        response = list_papers(limit=50, offset=0, source="arxiv")

        # Verify LIKE was used with arxiv% pattern
        calls = mock_cursor.execute.call_args_list
        # Count query should use LIKE
        count_call = calls[0]
        assert "LIKE" in count_call[0][0]
        assert count_call[0][1] == ["arxiv%"]

        # Select query should also use LIKE
        select_call = calls[1]
        assert "LIKE" in select_call[0][0]

        assert response.count == 2
        assert response.papers[0].source == "arxiv:cs.AI"
        assert response.papers[1].source == "arxiv:q-bio.NC"

    @patch("paperfind.api.get_conn")
    def test_crossref_source_uses_exact_match(self, mock_get_conn: MagicMock) -> None:
        """Filtering by 'crossref' should use exact match."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchone.return_value = {"total": 1}
        mock_cursor.fetchall.return_value = [
            {
                "doi": "10.1234/test",
                "title": "Paper",
                "authors": "Author",
                "abstract": "Abstract",
                "source": "crossref",
                "created_date": "2024-01-15",
            },
        ]
        mock_get_conn.return_value = mock_conn

        response = list_papers(limit=50, offset=0, source="crossref")

        # Verify exact match was used (= not LIKE)
        calls = mock_cursor.execute.call_args_list
        count_call = calls[0]
        assert "=" in count_call[0][0] and "LIKE" not in count_call[0][0]
        assert count_call[0][1] == ["crossref"]

        assert response.count == 1

    @patch("paperfind.api.get_conn")
    def test_biorxiv_source_uses_like_prefix(self, mock_get_conn: MagicMock) -> None:
        """Filtering by 'biorxiv' should use LIKE for potential sub-categories."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchone.return_value = {"total": 0}
        mock_cursor.fetchall.return_value = []
        mock_get_conn.return_value = mock_conn

        list_papers(limit=50, offset=0, source="biorxiv")

        calls = mock_cursor.execute.call_args_list
        count_call = calls[0]
        assert "LIKE" in count_call[0][0]
        assert count_call[0][1] == ["biorxiv%"]

    @patch("paperfind.api.get_conn")
    def test_no_source_filter_returns_all(self, mock_get_conn: MagicMock) -> None:
        """No source filter should return all papers."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchone.return_value = {"total": 10}
        mock_cursor.fetchall.return_value = []
        mock_get_conn.return_value = mock_conn

        list_papers(limit=50, offset=0, source=None)

        calls = mock_cursor.execute.call_args_list
        count_call = calls[0]
        # No WHERE clause
        assert "WHERE" not in count_call[0][0]


def test_fetch_accepts_chemrxiv_source() -> None:
    response = fetch(background_tasks=BackgroundTasks(), sources="chemrxiv")

    assert response.job_type == "fetch"
