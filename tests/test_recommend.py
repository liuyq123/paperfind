"""Tests for recommendation logic."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from paperfind.llm_rerank import llm_rerank_candidates
from paperfind.search.recommend import _strip_query, format_markdown, get_recommendations


class TestStripQuery:
    """Tests for _strip_query helper."""

    def test_strips_query_text(self):
        recommendations = [
            ("doi1", (0.1, Document(page_content="Doc1"), "Title1", "Query1")),
            ("doi2", (0.2, Document(page_content="Doc2"), "Title2", "Query2")),
        ]

        result = _strip_query(recommendations)

        assert len(result) == 2
        assert result[0] == ("doi1", (0.1, recommendations[0][1][1], "Title1"))
        assert result[1] == ("doi2", (0.2, recommendations[1][1][1], "Title2"))

    def test_empty_list(self):
        assert _strip_query([]) == []


class TestFormatMarkdown:
    """Tests for format_markdown."""

    def test_basic_markdown_output(self):
        doc = Document(
            page_content="Test Paper\n\nAbstract text.",
            metadata={
                "title": "Test Paper",
                "abstract": "Abstract text.",
                "authors": "A. Author",
                "source": "arxiv",
            },
        )
        recommendations = [("arxiv:2401.12345", (0.1, doc, "Seed Paper"))]

        result = format_markdown(recommendations, "2024-01-15")

        assert "# Paper Recommendations" in result
        assert "2024-01-15" in result
        assert "Test Paper" in result
        assert "Seed Paper" in result

    def test_with_collection(self):
        doc = Document(page_content="Test", metadata={"title": "Test"})
        recommendations = [("10.1234/test", (0.5, doc, "Seed"))]

        result = format_markdown(
            recommendations, "2024-01-15", collection="My Collection"
        )

        assert "My Collection" in result

    def test_empty_recommendations(self):
        result = format_markdown([], "2024-01-15")

        assert "# Paper Recommendations" in result
        assert "2024-01-15" in result

    def test_rerank_label(self):
        doc = Document(
            page_content="Test Paper\n\nAbstract text.",
            metadata={
                "title": "Test Paper",
                "abstract": "Abstract text.",
                "authors": "A. Author",
                "source": "arxiv",
            },
        )
        recommendations = [("arxiv:2401.12345", (9.5, doc, "Seed Paper"))]

        result = format_markdown(recommendations, "2024-01-15", rerank=True)

        assert "LLM score" in result


class TestGetRecommendations:
    """Tests for get_recommendations with mocked dependencies."""

    @patch("paperfind.search.recommend.check_vector_store")
    def test_returns_empty_when_no_vector_store(self, mock_check):
        mock_check.return_value = False

        result = get_recommendations(k=5)

        assert result == []

    @patch("paperfind.search.recommend.check_vector_store")
    @patch("paperfind.search.recommend.vector_store_exists")
    def test_returns_empty_when_no_zotero_embeddings(
        self, mock_store_exists, mock_check
    ):
        mock_check.return_value = True
        mock_store_exists.return_value = False

        result = get_recommendations(k=5)

        assert result == []

    @patch("paperfind.search.recommend.check_vector_store")
    @patch("paperfind.search.recommend.vector_store_exists")
    @patch("paperfind.search.recommend.get_zotero_papers")
    def test_returns_empty_when_no_zotero_papers(
        self, mock_get_papers, mock_store_exists, mock_check
    ):
        mock_check.return_value = True
        mock_store_exists.return_value = True
        mock_get_papers.return_value = []

        result = get_recommendations(k=5)

        assert result == []

    @patch("paperfind.search.recommend.check_vector_store")
    @patch("paperfind.search.recommend.vector_store_exists")
    @patch("paperfind.search.recommend.get_zotero_papers")
    @patch("paperfind.search.recommend.get_zotero_dois")
    @patch("paperfind.search.recommend.get_vector_store")
    @patch("paperfind.search.recommend.get_embeddings_from_store")
    @patch("paperfind.search.recommend.similarity_search_by_vector")
    @patch("paperfind.llm_rerank.llm_rerank_candidates")
    def test_rerank_path_returns_results(
        self,
        mock_llm_rerank,
        mock_search,
        mock_get_embeddings,
        mock_get_store,
        mock_get_dois,
        mock_get_papers,
        mock_store_exists,
        mock_check,
    ):
        mock_check.return_value = True
        mock_store_exists.return_value = True
        mock_get_papers.return_value = [
            {"zotero_key": "key1", "title": "Paper 1", "abstract": "Abstract 1"}
        ]
        mock_get_dois.return_value = set()
        mock_get_store.return_value = MagicMock()
        mock_get_embeddings.return_value = {"key1": [0.1, 0.2, 0.3]}

        doc = Document(
            page_content="Test",
            metadata={"doi": "10.1234/test", "title": "Test Paper"},
        )
        mock_search.return_value = [(doc, 0.5)]
        mock_llm_rerank.return_value = ([("10.1234/test", (9.5, doc, "Seed"))], True)

        result, rerank_used = get_recommendations(
            k=5,
            rerank=True,
            return_rerank_used=True,
        )

        assert rerank_used is True
        assert result[0][1][0] == 9.5

    @patch("paperfind.search.recommend.check_vector_store")
    @patch("paperfind.search.recommend.vector_store_exists")
    @patch("paperfind.search.recommend.get_zotero_papers")
    @patch("paperfind.search.recommend.get_zotero_dois")
    @patch("paperfind.search.recommend.get_vector_store")
    @patch("paperfind.search.recommend.get_embeddings_from_store")
    @patch("paperfind.search.recommend.similarity_search_by_vector")
    def test_filters_existing_dois(
        self,
        mock_search,
        mock_get_embeddings,
        mock_get_store,
        mock_get_dois,
        mock_get_papers,
        mock_store_exists,
        mock_check,
    ):
        # Setup mocks
        mock_check.return_value = True
        mock_store_exists.return_value = True
        mock_get_papers.return_value = [
            {"zotero_key": "key1", "title": "Paper 1", "abstract": "Abstract 1"}
        ]
        mock_get_dois.return_value = {"10.1234/existing"}
        mock_get_store.return_value = MagicMock()
        mock_get_embeddings.return_value = {"key1": [0.1, 0.2, 0.3]}

        # Mock search results including one that should be filtered
        doc_existing = Document(
            page_content="Existing",
            metadata={"doi": "10.1234/existing", "title": "Existing Paper"},
        )
        doc_new = Document(
            page_content="New",
            metadata={"doi": "10.1234/new", "title": "New Paper"},
        )
        mock_search.return_value = [(doc_existing, 0.1), (doc_new, 0.2)]

        result = get_recommendations(k=5, )

        # Should only contain the new paper, not the existing one
        assert len(result) == 1
        assert result[0][0] == "10.1234/new"

    @patch("paperfind.search.recommend.check_vector_store")
    @patch("paperfind.search.recommend.vector_store_exists")
    @patch("paperfind.search.recommend.get_zotero_papers")
    @patch("paperfind.search.recommend.get_zotero_dois")
    @patch("paperfind.search.recommend.get_vector_store")
    @patch("paperfind.search.recommend.get_embeddings_from_store")
    @patch("paperfind.search.recommend.similarity_search_by_vector")
    def test_exclude_dois_case_insensitive(
        self,
        mock_search,
        mock_get_embeddings,
        mock_get_store,
        mock_get_dois,
        mock_get_papers,
        mock_store_exists,
        mock_check,
    ):
        mock_check.return_value = True
        mock_store_exists.return_value = True
        mock_get_papers.return_value = [
            {"zotero_key": "key1", "title": "Paper 1", "abstract": "Abstract 1"}
        ]
        mock_get_dois.return_value = set()
        mock_get_store.return_value = MagicMock()
        mock_get_embeddings.return_value = {"key1": [0.1, 0.2, 0.3]}

        doc = Document(
            page_content="Test",
            metadata={"doi": "10.1234/new", "title": "New Paper"},
        )
        mock_search.return_value = [(doc, 0.5)]

        result = get_recommendations(
            k=5,
            exclude_dois={"10.1234/NEW"},
        )

        assert result == []

    @patch("paperfind.search.recommend.check_vector_store")
    @patch("paperfind.search.recommend.vector_store_exists")
    @patch("paperfind.search.recommend.get_zotero_papers")
    @patch("paperfind.search.recommend.get_zotero_dois")
    @patch("paperfind.search.recommend.get_vector_store")
    @patch("paperfind.search.recommend.get_embeddings_from_store")
    @patch("paperfind.search.recommend.similarity_search_by_vector")
    def test_returns_rerank_used_flag(
        self,
        mock_search,
        mock_get_embeddings,
        mock_get_store,
        mock_get_dois,
        mock_get_papers,
        mock_store_exists,
        mock_check,
    ):
        mock_check.return_value = True
        mock_store_exists.return_value = True
        mock_get_papers.return_value = [
            {"zotero_key": "key1", "title": "Paper 1", "abstract": "Abstract 1"}
        ]
        mock_get_dois.return_value = set()
        mock_get_store.return_value = MagicMock()
        mock_get_embeddings.return_value = {"key1": [0.1, 0.2, 0.3]}

        doc = Document(
            page_content="Test",
            metadata={"doi": "10.1234/test", "title": "Test Paper"},
        )
        mock_search.return_value = [(doc, 0.5)]

        # Without LLM rerank
        result, rerank_used = get_recommendations(
            k=5, return_rerank_used=True
        )

        assert rerank_used is False
        assert len(result) == 1

    @patch("paperfind.search.recommend.check_vector_store")
    @patch("paperfind.search.recommend.vector_store_exists")
    @patch("paperfind.search.recommend.get_zotero_papers")
    @patch("paperfind.search.recommend.get_zotero_dois")
    @patch("paperfind.search.recommend.get_vector_store")
    @patch("paperfind.search.recommend.get_embeddings_from_store")
    @patch("paperfind.search.recommend.similarity_search_by_vector")
    def test_keeps_best_score_per_doi(
        self,
        mock_search,
        mock_get_embeddings,
        mock_get_store,
        mock_get_dois,
        mock_get_papers,
        mock_store_exists,
        mock_check,
    ):
        mock_check.return_value = True
        mock_store_exists.return_value = True
        mock_get_papers.return_value = [
            {"zotero_key": "key1", "title": "Paper 1", "abstract": "Abstract 1"},
            {"zotero_key": "key2", "title": "Paper 2", "abstract": "Abstract 2"},
        ]
        mock_get_dois.return_value = set()
        mock_get_store.return_value = MagicMock()
        mock_get_embeddings.return_value = {
            "key1": [0.1, 0.2, 0.3],
            "key2": [0.4, 0.5, 0.6],
        }

        # Same DOI returned by both queries with different scores
        doc = Document(
            page_content="Test",
            metadata={"doi": "10.1234/same", "title": "Same Paper"},
        )
        # First call returns score 0.5, second returns 0.2 (lower is better)
        mock_search.side_effect = [
            [(doc, 0.5)],
            [(doc, 0.2)],
        ]

        result = get_recommendations(k=5, )

        # Should keep the better (lower) score
        assert len(result) == 1
        assert result[0][0] == "10.1234/same"
        assert result[0][1][0] == 0.2  # Lower score

    @patch("paperfind.search.recommend.check_vector_store")
    @patch("paperfind.search.recommend.vector_store_exists")
    @patch("paperfind.search.recommend.get_zotero_papers")
    @patch("paperfind.search.recommend.get_zotero_dois")
    @patch("paperfind.search.recommend.get_vector_store")
    @patch("paperfind.search.recommend.get_embeddings_from_store")
    def test_returns_empty_when_no_embeddings_found(
        self,
        mock_get_embeddings,
        mock_get_store,
        mock_get_dois,
        mock_get_papers,
        mock_store_exists,
        mock_check,
    ):
        mock_check.return_value = True
        mock_store_exists.return_value = True
        mock_get_papers.return_value = [
            {"zotero_key": "key1", "title": "Paper 1", "abstract": "Abstract 1"}
        ]
        mock_get_dois.return_value = set()
        mock_get_store.return_value = MagicMock()
        mock_get_embeddings.return_value = {}  # No embeddings found

        result = get_recommendations(k=5, )

        assert result == []

    @patch("paperfind.search.recommend.check_vector_store")
    @patch("paperfind.search.recommend.vector_store_exists")
    @patch("paperfind.search.recommend.get_zotero_papers")
    @patch("paperfind.search.recommend.get_zotero_dois")
    @patch("paperfind.search.recommend.get_vector_store")
    @patch("paperfind.search.recommend.get_embeddings_from_store")
    @patch("paperfind.search.recommend.similarity_search_by_vector")
    def test_skips_docs_without_doi(
        self,
        mock_search,
        mock_get_embeddings,
        mock_get_store,
        mock_get_dois,
        mock_get_papers,
        mock_store_exists,
        mock_check,
    ):
        mock_check.return_value = True
        mock_store_exists.return_value = True
        mock_get_papers.return_value = [
            {"zotero_key": "key1", "title": "Paper 1", "abstract": "Abstract 1"}
        ]
        mock_get_dois.return_value = set()
        mock_get_store.return_value = MagicMock()
        mock_get_embeddings.return_value = {"key1": [0.1, 0.2, 0.3]}

        # One doc without DOI, one with
        doc_no_doi = Document(page_content="No DOI", metadata={"title": "No DOI Paper"})
        doc_with_doi = Document(
            page_content="With DOI",
            metadata={"doi": "10.1234/test", "title": "With DOI Paper"},
        )
        mock_search.return_value = [(doc_no_doi, 0.1), (doc_with_doi, 0.2)]

        result = get_recommendations(k=5, )

        assert len(result) == 1
        assert result[0][0] == "10.1234/test"


class TestExcludeDois:
    """Tests for exclude_dois parameter."""

    @patch("paperfind.search.recommend.check_vector_store")
    @patch("paperfind.search.recommend.vector_store_exists")
    @patch("paperfind.search.recommend.get_zotero_papers")
    @patch("paperfind.search.recommend.get_zotero_dois")
    @patch("paperfind.search.recommend.get_vector_store")
    @patch("paperfind.search.recommend.get_embeddings_from_store")
    @patch("paperfind.search.recommend.similarity_search_by_vector")
    def test_excludes_specified_dois(
        self,
        mock_search,
        mock_get_embeddings,
        mock_get_store,
        mock_get_dois,
        mock_get_papers,
        mock_store_exists,
        mock_check,
    ):
        mock_check.return_value = True
        mock_store_exists.return_value = True
        mock_get_papers.return_value = [
            {"zotero_key": "key1", "title": "Paper 1", "abstract": "Abstract 1"}
        ]
        mock_get_dois.return_value = set()
        mock_get_store.return_value = MagicMock()
        mock_get_embeddings.return_value = {"key1": [0.1, 0.2, 0.3]}

        # Three docs, one should be excluded
        doc1 = Document(
            page_content="First",
            metadata={"doi": "10.1234/first", "title": "First Paper"},
        )
        doc2 = Document(
            page_content="Second",
            metadata={"doi": "10.1234/excluded", "title": "Excluded Paper"},
        )
        doc3 = Document(
            page_content="Third",
            metadata={"doi": "10.1234/third", "title": "Third Paper"},
        )
        mock_search.return_value = [(doc1, 0.1), (doc2, 0.2), (doc3, 0.3)]

        result = get_recommendations(
            k=5, exclude_dois={"10.1234/excluded"}
        )

        # Should exclude the specified DOI
        assert len(result) == 2
        dois = [r[0] for r in result]
        assert "10.1234/first" in dois
        assert "10.1234/third" in dois
        assert "10.1234/excluded" not in dois

    @patch("paperfind.search.recommend.check_vector_store")
    @patch("paperfind.search.recommend.vector_store_exists")
    @patch("paperfind.search.recommend.get_zotero_papers")
    @patch("paperfind.search.recommend.get_zotero_dois")
    @patch("paperfind.search.recommend.get_vector_store")
    @patch("paperfind.search.recommend.get_embeddings_from_store")
    @patch("paperfind.search.recommend.similarity_search_by_vector")
    def test_empty_exclude_dois_returns_all(
        self,
        mock_search,
        mock_get_embeddings,
        mock_get_store,
        mock_get_dois,
        mock_get_papers,
        mock_store_exists,
        mock_check,
    ):
        mock_check.return_value = True
        mock_store_exists.return_value = True
        mock_get_papers.return_value = [
            {"zotero_key": "key1", "title": "Paper 1", "abstract": "Abstract 1"}
        ]
        mock_get_dois.return_value = set()
        mock_get_store.return_value = MagicMock()
        mock_get_embeddings.return_value = {"key1": [0.1, 0.2, 0.3]}

        doc = Document(
            page_content="Test",
            metadata={"doi": "10.1234/test", "title": "Test Paper"},
        )
        mock_search.return_value = [(doc, 0.5)]

        # Empty exclude set should not filter anything
        result = get_recommendations(k=5, exclude_dois=set())

        assert len(result) == 1


class TestLLMRerankCandidates:
    """Tests for LLM rerank fallback behavior."""

    def test_falls_back_on_error(self):
        doc = Document(
            page_content="Test",
            metadata={"doi": "10.1234/test", "title": "Test Paper"},
        )
        candidates = [("10.1234/test", (0.5, doc, "Seed", "query"))]

        with patch("paperfind.llm_rerank.llm_rerank", side_effect=RuntimeError("boom")):
            results, used = llm_rerank_candidates(candidates, k=1)

        assert used is False
        assert results[0][0] == "10.1234/test"
        assert results[0][1][0] == 0.5

    @patch("paperfind.search.recommend.check_vector_store")
    @patch("paperfind.search.recommend.vector_store_exists")
    @patch("paperfind.search.recommend.get_zotero_papers")
    @patch("paperfind.search.recommend.get_zotero_dois")
    @patch("paperfind.search.recommend.get_vector_store")
    @patch("paperfind.search.recommend.get_embeddings_from_store")
    @patch("paperfind.search.recommend.similarity_search_by_vector")
    def test_none_exclude_dois_returns_all(
        self,
        mock_search,
        mock_get_embeddings,
        mock_get_store,
        mock_get_dois,
        mock_get_papers,
        mock_store_exists,
        mock_check,
    ):
        mock_check.return_value = True
        mock_store_exists.return_value = True
        mock_get_papers.return_value = [
            {"zotero_key": "key1", "title": "Paper 1", "abstract": "Abstract 1"}
        ]
        mock_get_dois.return_value = set()
        mock_get_store.return_value = MagicMock()
        mock_get_embeddings.return_value = {"key1": [0.1, 0.2, 0.3]}

        doc = Document(
            page_content="Test",
            metadata={"doi": "10.1234/test", "title": "Test Paper"},
        )
        mock_search.return_value = [(doc, 0.5)]

        # None exclude set should not filter anything
        result = get_recommendations(k=5, exclude_dois=None)

        assert len(result) == 1
