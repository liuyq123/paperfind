"""Tests for search functionality."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from paperfind.search.search import search, search_with_scores


class TestSearch:
    """Tests for the search function."""

    @patch("paperfind.search.search.get_vectordb")
    @patch("paperfind.search.search.warn_if_empty")
    def test_basic_search(self, mock_warn, mock_get_db):
        mock_db = MagicMock()
        doc = Document(page_content="Test Paper", metadata={"title": "Test"})
        mock_db.similarity_search.return_value = [doc]
        mock_get_db.return_value = mock_db

        results = search("machine learning", k=5)

        mock_db.similarity_search.assert_called_once_with("machine learning", k=5)
        assert len(results) == 1
        assert results[0] == doc

    @patch("paperfind.search.search.get_vectordb")
    @patch("paperfind.search.search.warn_if_empty")
    def test_search_with_source(self, mock_warn, mock_get_db):
        mock_db = MagicMock()
        mock_db.similarity_search.return_value = []
        mock_get_db.return_value = mock_db

        search("query", k=3, source="zotero")

        mock_get_db.assert_called_once_with("zotero")

    @patch("paperfind.search.search.get_vectordb")
    @patch("paperfind.search.search.warn_if_empty")
    @patch("paperfind.search.search.get_collection_zotero_keys")
    def test_search_with_collection_filter(
        self, mock_get_keys, mock_warn, mock_get_db
    ):
        mock_db = MagicMock()
        # Return docs with different zotero_keys
        docs = [
            Document(
                page_content="Paper 1",
                metadata={"title": "Paper 1", "zotero_key": "key1"},
            ),
            Document(
                page_content="Paper 2",
                metadata={"title": "Paper 2", "zotero_key": "key2"},
            ),
            Document(
                page_content="Paper 3",
                metadata={"title": "Paper 3", "zotero_key": "key3"},
            ),
        ]
        mock_db.similarity_search.return_value = docs
        mock_get_db.return_value = mock_db

        # Only allow key1 and key3
        mock_get_keys.return_value = {"key1", "key3"}

        results = search("query", k=5, source="zotero", collection="my-collection")

        # Should filter to only key1 and key3
        assert len(results) == 2
        assert results[0].metadata["zotero_key"] == "key1"
        assert results[1].metadata["zotero_key"] == "key3"

    @patch("paperfind.search.search.get_vectordb")
    @patch("paperfind.search.search.warn_if_empty")
    @patch("paperfind.search.search.get_collection_zotero_keys")
    def test_search_with_empty_collection(self, mock_get_keys, mock_warn, mock_get_db):
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_get_keys.return_value = set()  # Empty collection

        results = search("query", k=5, source="zotero", collection="empty-collection")

        assert results == []
        mock_db.similarity_search.assert_not_called()

    @patch("paperfind.search.search.get_vectordb")
    @patch("paperfind.search.search.warn_if_empty")
    def test_search_increases_k_for_collection_filter(self, mock_warn, mock_get_db):
        mock_db = MagicMock()
        mock_db.similarity_search.return_value = []
        mock_get_db.return_value = mock_db

        with patch(
            "paperfind.search.search.get_collection_zotero_keys"
        ) as mock_get_keys:
            mock_get_keys.return_value = {"key1"}

            search("query", k=5, source="zotero", collection="test")

            # Should search with k * 3 when filtering
            mock_db.similarity_search.assert_called_once_with("query", k=15)


class TestSearchWithScores:
    """Tests for search_with_scores."""

    @patch("paperfind.search.search.get_vectordb")
    @patch("paperfind.search.search.warn_if_empty")
    def test_returns_documents_with_scores(self, mock_warn, mock_get_db):
        mock_db = MagicMock()
        doc = Document(page_content="Test", metadata={"title": "Test"})
        mock_db.similarity_search_with_score.return_value = [(doc, 0.85)]
        mock_get_db.return_value = mock_db

        results = search_with_scores("query", k=5)

        assert len(results) == 1
        assert results[0][0] == doc
        assert results[0][1] == 0.85

    @patch("paperfind.search.search.get_vectordb")
    @patch("paperfind.search.search.warn_if_empty")
    def test_uses_correct_source(self, mock_warn, mock_get_db):
        mock_db = MagicMock()
        mock_db.similarity_search_with_score.return_value = []
        mock_get_db.return_value = mock_db

        search_with_scores("query", k=3, source="daily_papers")

        mock_get_db.assert_called_once_with("daily_papers")

    @patch("paperfind.search.search.get_vectordb")
    @patch("paperfind.search.search.warn_if_empty")
    @patch("paperfind.search.search.get_collection_zotero_keys")
    def test_search_with_scores_collection_filter(
        self, mock_get_keys, mock_warn, mock_get_db
    ):
        mock_db = MagicMock()
        docs_with_scores = [
            (Document(page_content="Paper 1", metadata={"zotero_key": "key1"}), 0.1),
            (Document(page_content="Paper 2", metadata={"zotero_key": "key2"}), 0.2),
            (Document(page_content="Paper 3", metadata={"zotero_key": "key3"}), 0.3),
        ]
        mock_db.similarity_search_with_score.return_value = docs_with_scores
        mock_get_db.return_value = mock_db
        mock_get_keys.return_value = {"key1", "key3"}

        results = search_with_scores(
            "query",
            k=2,
            source="zotero",
            collection="my-collection",
        )

        mock_db.similarity_search_with_score.assert_called_once_with("query", k=6)
        assert len(results) == 2
        assert results[0][0].metadata["zotero_key"] == "key1"
        assert results[1][0].metadata["zotero_key"] == "key3"

    @patch("paperfind.search.search.get_vectordb")
    @patch("paperfind.search.search.warn_if_empty")
    @patch("paperfind.search.search.get_collection_zotero_keys")
    def test_search_with_scores_empty_collection(
        self, mock_get_keys, mock_warn, mock_get_db
    ):
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db
        mock_get_keys.return_value = set()

        results = search_with_scores("query", k=5, source="zotero", collection="empty")

        assert results == []
        mock_db.similarity_search_with_score.assert_not_called()


class TestSearchUtils:
    """Tests for search utility functions."""

    @patch("paperfind.search.search.get_db")
    def test_get_collection_zotero_keys_empty(self, mock_get_db):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None  # Collection not found
        mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_db.return_value.__exit__ = MagicMock(return_value=False)

        from paperfind.search.search import get_collection_zotero_keys

        keys = get_collection_zotero_keys("nonexistent")

        assert keys == set()

    @patch("paperfind.search.search.get_db")
    def test_get_collection_zotero_keys_found(self, mock_get_db):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor

        # First call: find collection
        # Second call: get keys
        mock_cursor.fetchone.return_value = {"id": 1}
        mock_cursor.fetchall.return_value = [
            {"zotero_key": "key1"},
            {"zotero_key": "key2"},
        ]
        mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_db.return_value.__exit__ = MagicMock(return_value=False)

        from paperfind.search.search import get_collection_zotero_keys

        keys = get_collection_zotero_keys("my-collection")

        assert keys == {"key1", "key2"}
