"""Tests for reranking utilities."""

from unittest.mock import MagicMock, patch

import pytest

from paperfind.rerank import (
    DEFAULT_RERANK_MODEL,
    get_rerank_model,
    rerank_pairs,
)


class TestGetRerankModel:
    """Tests for get_rerank_model."""

    def test_default_model(self):
        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("RERANK_MODEL", None)
            assert get_rerank_model() == DEFAULT_RERANK_MODEL

    def test_custom_model_from_env(self):
        with patch.dict("os.environ", {"RERANK_MODEL": "custom-model"}):
            assert get_rerank_model() == "custom-model"


class TestRerankPairs:
    """Tests for rerank_pairs."""

    def test_empty_pairs_returns_empty(self):
        result = rerank_pairs([])
        assert result == []

    @patch("paperfind.rerank._get_cross_encoder")
    def test_uses_cross_encoder(self, mock_get_encoder):
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [0.8, 0.6]
        mock_get_encoder.return_value = mock_encoder

        pairs = [("query1", "doc1"), ("query2", "doc2")]
        result = rerank_pairs(pairs)

        mock_encoder.predict.assert_called_once_with(pairs)
        assert result == [0.8, 0.6]

    @patch("paperfind.rerank._get_cross_encoder")
    def test_converts_numpy_array_to_list(self, mock_get_encoder):
        import numpy as np

        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = np.array([0.9, 0.5, 0.3])
        mock_get_encoder.return_value = mock_encoder

        pairs = [("q1", "d1"), ("q2", "d2"), ("q3", "d3")]
        result = rerank_pairs(pairs)

        assert isinstance(result, list)
        assert result == [0.9, 0.5, 0.3]

    @patch("paperfind.rerank._get_cross_encoder")
    def test_uses_custom_model(self, mock_get_encoder):
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [0.7]
        mock_get_encoder.return_value = mock_encoder

        pairs = [("query", "doc")]
        rerank_pairs(pairs, model="my-custom-model")

        mock_get_encoder.assert_called_once_with("my-custom-model")

    @patch("paperfind.rerank._get_cross_encoder")
    def test_uses_default_model_when_none(self, mock_get_encoder):
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [0.7]
        mock_get_encoder.return_value = mock_encoder

        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("RERANK_MODEL", None)
            pairs = [("query", "doc")]
            rerank_pairs(pairs, model=None)

            mock_get_encoder.assert_called_once_with(DEFAULT_RERANK_MODEL)


class TestCrossEncoderImport:
    """Tests for cross-encoder import handling."""

    def test_import_error_raised_when_not_installed(self):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            from paperfind import rerank

            # Clear the cache
            rerank._get_cross_encoder.cache_clear()

            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'sentence_transformers'"),
            ):
                with pytest.raises(ImportError) as exc_info:
                    rerank._get_cross_encoder("test-model")

                assert "sentence-transformers" in str(exc_info.value)
