"""Tests for vectorstore utility functions."""

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from paperfind.vectorstore import (
    _build_filter_clause,
    _normalize_metadata,
    _sanitize_identifier,
    _truncate_identifier,
    get_embeddings_from_store,
    get_vector_store_backend,
    similarity_search_by_vector,
)


class TestSanitizeIdentifier:
    """Tests for _sanitize_identifier."""

    def test_simple_name(self):
        assert _sanitize_identifier("openai") == "openai"

    def test_with_slashes(self):
        # Slashes become underscores
        result = _sanitize_identifier("text-embedding-3-small")
        assert "/" not in result
        assert result.islower()

    def test_with_special_chars(self):
        result = _sanitize_identifier("model@v1.0")
        assert "@" not in result
        assert "." not in result

    def test_empty_string(self):
        assert _sanitize_identifier("") == "default"

    def test_only_special_chars(self):
        assert _sanitize_identifier("@#$%") == "default"

    def test_mixed_case(self):
        result = _sanitize_identifier("OpenAI_Model")
        assert result == result.lower()


class TestTruncateIdentifier:
    """Tests for _truncate_identifier."""

    def test_short_name_unchanged(self):
        name = "short_name"
        assert _truncate_identifier(name) == name

    def test_exact_max_length(self):
        name = "a" * 63
        assert _truncate_identifier(name) == name

    def test_long_name_truncated(self):
        name = "a" * 100
        result = _truncate_identifier(name)
        assert len(result) <= 63
        # Should end with hash
        assert "_" in result

    def test_custom_max_length(self):
        name = "a" * 30
        result = _truncate_identifier(name, max_len=20)
        assert len(result) <= 20


class TestNormalizeMetadata:
    """Tests for _normalize_metadata."""

    def test_none_returns_empty_dict(self):
        assert _normalize_metadata(None) == {}

    def test_dict_returned_as_is(self):
        data = {"key": "value"}
        assert _normalize_metadata(data) == data

    def test_json_string_parsed(self):
        data = {"title": "Test", "authors": "A. Author"}
        json_str = json.dumps(data)
        assert _normalize_metadata(json_str) == data

    def test_invalid_json_returns_empty(self):
        assert _normalize_metadata("not valid json") == {}

    def test_empty_string_returns_empty(self):
        assert _normalize_metadata("") == {}


class TestBuildFilterClause:
    """Tests for _build_filter_clause."""

    def test_none_filter(self):
        clause, params = _build_filter_clause(None)
        assert clause == ""
        assert params == []

    def test_empty_filter(self):
        clause, params = _build_filter_clause({})
        assert clause == ""
        assert params == []

    def test_simple_filter(self):
        clause, params = _build_filter_clause({"source": "arxiv"})
        assert "WHERE" in clause
        assert "metadata @>" in clause
        assert len(params) == 1


class TestGetVectorStoreBackend:
    """Tests for get_vector_store_backend."""

    def test_default_is_chroma(self):
        with patch.dict("os.environ", {}, clear=True):
            # Clear any vector store env vars
            import os
            for key in ["PAPERFIND_VECTOR_STORE", "VECTOR_STORE"]:
                os.environ.pop(key, None)
            assert get_vector_store_backend() == "chroma"

    def test_pgvector_from_env(self):
        with patch.dict("os.environ", {"PAPERFIND_VECTOR_STORE": "pgvector"}):
            assert get_vector_store_backend() == "pgvector"

    def test_case_insensitive(self):
        with patch.dict("os.environ", {"PAPERFIND_VECTOR_STORE": "PGVECTOR"}):
            assert get_vector_store_backend() == "pgvector"

    def test_unknown_falls_back_to_default(self):
        with patch.dict("os.environ", {"PAPERFIND_VECTOR_STORE": "unknown_store"}):
            assert get_vector_store_backend() == "chroma"


class TestGetEmbeddingsFromStore:
    """Tests for get_embeddings_from_store."""

    def test_empty_ids_returns_empty(self):
        mock_store = MagicMock()
        result = get_embeddings_from_store(mock_store, [])
        assert result == {}

    def test_pgvector_store_uses_native_method(self):
        from paperfind.vectorstore import PGVectorStore

        mock_store = MagicMock(spec=PGVectorStore)
        mock_store.get_embeddings_by_ids.return_value = {"id1": [0.1, 0.2]}

        result = get_embeddings_from_store(mock_store, ["id1"])

        mock_store.get_embeddings_by_ids.assert_called_once_with(["id1"])
        assert result == {"id1": [0.1, 0.2]}

    def test_chroma_store_uses_collection(self):
        mock_store = MagicMock()
        mock_collection = MagicMock()
        mock_store._collection = mock_collection
        mock_collection.get.return_value = {
            "ids": ["id1", "id2"],
            "embeddings": [[0.1, 0.2], [0.3, 0.4]],
        }

        result = get_embeddings_from_store(mock_store, ["id1", "id2"])

        mock_collection.get.assert_called_once_with(
            ids=["id1", "id2"], include=["embeddings"]
        )
        assert result == {"id1": [0.1, 0.2], "id2": [0.3, 0.4]}

    def test_chroma_empty_result(self):
        mock_store = MagicMock()
        mock_collection = MagicMock()
        mock_store._collection = mock_collection
        mock_collection.get.return_value = {"ids": [], "embeddings": []}

        result = get_embeddings_from_store(mock_store, ["id1"])

        assert result == {}

    def test_chroma_exception_returns_empty(self):
        mock_store = MagicMock()
        mock_collection = MagicMock()
        mock_store._collection = mock_collection
        mock_collection.get.side_effect = Exception("Collection error")

        result = get_embeddings_from_store(mock_store, ["id1"])

        assert result == {}

    def test_unknown_store_returns_empty(self):
        mock_store = MagicMock(spec=[])  # No _collection attribute

        result = get_embeddings_from_store(mock_store, ["id1"])

        assert result == {}


class TestSimilaritySearchByVector:
    """Tests for similarity_search_by_vector."""

    def test_pgvector_store_uses_native_method(self):
        from paperfind.vectorstore import PGVectorStore

        mock_store = MagicMock(spec=PGVectorStore)
        doc = Document(page_content="Test", metadata={})
        mock_store.similarity_search_by_vector_with_score.return_value = [(doc, 0.5)]

        result = similarity_search_by_vector(mock_store, [0.1, 0.2], k=5)

        mock_store.similarity_search_by_vector_with_score.assert_called_once_with(
            [0.1, 0.2], k=5
        )
        assert result == [(doc, 0.5)]

    def test_langchain_store_with_relevance_scores(self):
        mock_store = MagicMock()
        doc = Document(page_content="Test", metadata={})
        mock_store.similarity_search_by_vector_with_relevance_scores.return_value = [
            (doc, 0.2)
        ]

        result = similarity_search_by_vector(mock_store, [0.1, 0.2], k=3)

        mock_store.similarity_search_by_vector_with_relevance_scores.assert_called_once()
        # Chroma returns distance-like scores (lower = more similar), passed through as-is
        assert result == [(doc, 0.2)]

    def test_scores_passed_through_as_distance(self):
        """Chroma returns distance-like scores (lower = more similar)."""
        mock_store = MagicMock()
        doc_best = Document(page_content="Best", metadata={})
        doc_worse = Document(page_content="Worse", metadata={})
        # Lower score = more similar (Chroma's behavior)
        mock_store.similarity_search_by_vector_with_relevance_scores.return_value = [
            (doc_best, 0.1),
            (doc_worse, 0.9),
        ]

        result = similarity_search_by_vector(mock_store, [0.1, 0.2], k=2)

        # Scores passed through unchanged
        assert result == [
            (doc_best, 0.1),
            (doc_worse, 0.9),
        ]
        assert result[0][1] < result[1][1]  # Best (lower) < Worse (higher)

    def test_langchain_store_without_scores_fallback(self):
        mock_store = MagicMock(spec=["similarity_search_by_vector"])
        doc = Document(page_content="Test", metadata={})
        mock_store.similarity_search_by_vector.return_value = [doc]

        result = similarity_search_by_vector(mock_store, [0.1, 0.2], k=3)

        mock_store.similarity_search_by_vector.assert_called_once()
        assert result == [(doc, 0.0)]

    def test_unsupported_store_raises(self):
        mock_store = MagicMock(spec=[])  # No search methods

        with pytest.raises(NotImplementedError):
            similarity_search_by_vector(mock_store, [0.1, 0.2], k=3)

    def test_chroma_backend_no_warning(self, caplog):
        """Chroma backend should not warn about score direction."""

        class MockChroma:
            def similarity_search_by_vector_with_relevance_scores(self, embedding, k, **kwargs):
                doc = Document(page_content="Test", metadata={})
                return [(doc, 0.5)]  # Score in [0, 1] range

        result = similarity_search_by_vector(MockChroma(), [0.1, 0.2], k=3)

        assert len(result) == 1
        # No warning because "Chroma" is in class name
        assert "score direction" not in caplog.text.lower()

    def test_unknown_backend_warns_on_relevance_scores(self, caplog):
        """Unknown backend with [0,1] scores should warn about potential inversion."""
        import logging

        class UnknownVectorStore:
            def similarity_search_by_vector_with_relevance_scores(self, embedding, k, **kwargs):
                doc = Document(page_content="Test", metadata={})
                return [(doc, 0.8)]  # Score in [0, 1] range - might be relevance!

        with caplog.at_level(logging.WARNING):
            result = similarity_search_by_vector(UnknownVectorStore(), [0.1, 0.2], k=3)

        assert len(result) == 1
        # Should warn because unknown backend returned [0,1] scores
        assert "ranking may be inverted" in caplog.text

    def test_unknown_backend_no_warning_on_distance_scores(self, caplog):
        """Unknown backend with scores > 1 should not warn (clearly distance)."""

        class UnknownVectorStore:
            def similarity_search_by_vector_with_relevance_scores(self, embedding, k, **kwargs):
                doc = Document(page_content="Test", metadata={})
                return [(doc, 1.5)]  # Score > 1, clearly distance

        result = similarity_search_by_vector(UnknownVectorStore(), [0.1, 0.2], k=3)

        assert len(result) == 1
        # No warning because score > 1 indicates distance
        assert "ranking may be inverted" not in caplog.text
