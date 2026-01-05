"""Tests for paper fetcher modules."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest
import requests


class TestArxivFetcher:
    """Tests for arXiv paper fetcher."""

    @pytest.fixture
    def sample_arxiv_xml(self):
        """Sample arXiv API response XML."""
        return """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom"
              xmlns:arxiv="http://arxiv.org/schemas/atom">
            <entry>
                <id>http://arxiv.org/abs/2401.12345v1</id>
                <title>Test Paper Title</title>
                <summary>This is the abstract of the test paper.</summary>
                <author><name>John Doe</name></author>
                <author><name>Jane Smith</name></author>
                <published>2024-01-15T00:00:00Z</published>
            </entry>
            <entry>
                <id>http://arxiv.org/abs/2401.12346v2</id>
                <title>Another Paper</title>
                <summary>Another abstract.</summary>
                <author><name>Bob Wilson</name></author>
                <published>2024-01-14T00:00:00Z</published>
            </entry>
        </feed>"""

    @patch("paperfind.fetchers.sources.arxiv.requests.get")
    def test_fetch_arxiv_success(self, mock_get, sample_arxiv_xml):
        """Test successful fetch from arXiv."""
        from paperfind.fetchers.sources.arxiv import fetch_arxiv

        mock_response = MagicMock()
        mock_response.content = sample_arxiv_xml.encode()
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with patch("paperfind.fetchers.sources.arxiv.date") as mock_date:
            mock_date.today.return_value = date(2024, 1, 16)
            mock_date.fromisoformat = date.fromisoformat
            papers = fetch_arxiv("cs.AI", days=7)

        assert len(papers) == 2
        assert papers[0]["doi"] == "arxiv:2401.12345"
        assert papers[0]["title"] == "Test Paper Title"
        assert papers[0]["authors"] == "John Doe, Jane Smith"
        assert papers[0]["abstract"] == "This is the abstract of the test paper."
        assert papers[0]["source"] == "arxiv:cs.AI"
        assert papers[0]["type"] == "preprint"

    @patch("paperfind.fetchers.sources.arxiv.requests.get")
    def test_fetch_arxiv_empty_response(self, mock_get):
        """Test empty response from arXiv."""
        from paperfind.fetchers.sources.arxiv import fetch_arxiv

        empty_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom"></feed>"""

        mock_response = MagicMock()
        mock_response.content = empty_xml.encode()
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        papers = fetch_arxiv("cs.AI", days=1)
        assert papers == []

    @patch("paperfind.fetchers.sources.arxiv.requests.get")
    def test_fetch_arxiv_network_error(self, mock_get):
        """Test network error handling."""
        from paperfind.fetchers.sources.arxiv import fetch_arxiv

        mock_get.side_effect = requests.RequestException("Connection failed")

        papers = fetch_arxiv("cs.AI", days=1)
        assert papers == []

    @patch("paperfind.fetchers.sources.arxiv.requests.get")
    def test_fetch_arxiv_malformed_xml(self, mock_get):
        """Test malformed XML handling."""
        from paperfind.fetchers.sources.arxiv import fetch_arxiv

        mock_response = MagicMock()
        mock_response.content = b"not valid xml"
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        papers = fetch_arxiv("cs.AI", days=1)
        assert papers == []

    @patch("paperfind.fetchers.sources.arxiv.requests.get")
    def test_fetch_arxiv_skips_entries_without_required_fields(self, mock_get):
        """Test that entries without required fields are skipped."""
        from paperfind.fetchers.sources.arxiv import fetch_arxiv

        xml_missing_abstract = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/2401.12345v1</id>
                <title>Test Paper</title>
                <published>2024-01-15T00:00:00Z</published>
            </entry>
        </feed>"""

        mock_response = MagicMock()
        mock_response.content = xml_missing_abstract.encode()
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        with patch("paperfind.fetchers.sources.arxiv.date") as mock_date:
            mock_date.today.return_value = date(2024, 1, 16)
            mock_date.fromisoformat = date.fromisoformat
            papers = fetch_arxiv("cs.AI", days=7)

        assert papers == []

    @patch("paperfind.fetchers.sources.arxiv.requests.get")
    def test_fetch_arxiv_date_filtering(self, mock_get, sample_arxiv_xml):
        """Test that old papers are filtered out."""
        from paperfind.fetchers.sources.arxiv import fetch_arxiv

        mock_response = MagicMock()
        mock_response.content = sample_arxiv_xml.encode()
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        # Set today to be far in the future so papers are filtered out
        with patch("paperfind.fetchers.sources.arxiv.date") as mock_date:
            mock_date.today.return_value = date(2025, 1, 1)
            mock_date.fromisoformat = date.fromisoformat
            papers = fetch_arxiv("cs.AI", days=1)

        assert papers == []


class TestBiorxivFetcher:
    """Tests for bioRxiv/medRxiv paper fetcher."""

    @pytest.fixture
    def sample_biorxiv_response(self):
        """Sample bioRxiv API response."""
        return {
            "collection": [
                {
                    "doi": "10.1101/2024.01.15.123456",
                    "title": "A Bioinformatics Study",
                    "authors": "John Doe, Jane Smith",
                    "abstract": "This is a test abstract.",
                    "date": "2024-01-15",
                    "category": "bioinformatics",
                },
                {
                    "doi": "10.1101/2024.01.14.789012",
                    "title": "Another Study",
                    "authors": "Bob Wilson",
                    "abstract": "Another abstract here.",
                    "date": "2024-01-14",
                    "category": "genomics",
                },
            ]
        }

    @patch("paperfind.fetchers.sources.biorxiv.requests.get")
    def test_fetch_biorxiv_success(self, mock_get, sample_biorxiv_response):
        """Test successful fetch from bioRxiv."""
        from paperfind.fetchers.sources.biorxiv import fetch_biorxiv

        mock_response = MagicMock()
        mock_response.json.return_value = sample_biorxiv_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        start_date = date(2024, 1, 14)
        end_date = date(2024, 1, 15)
        papers = fetch_biorxiv(start_date, end_date)

        assert len(papers) == 2
        assert papers[0]["doi"] == "10.1101/2024.01.15.123456"
        assert papers[0]["title"] == "A Bioinformatics Study"
        assert papers[0]["source"] == "biorxiv"
        assert papers[0]["type"] == "preprint"

    @patch("paperfind.fetchers.sources.biorxiv.requests.get")
    def test_fetch_biorxiv_with_category_filter(self, mock_get, sample_biorxiv_response):
        """Test category filtering."""
        from paperfind.fetchers.sources.biorxiv import fetch_biorxiv

        mock_response = MagicMock()
        mock_response.json.return_value = sample_biorxiv_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        start_date = date(2024, 1, 14)
        end_date = date(2024, 1, 15)
        papers = fetch_biorxiv(start_date, end_date, category="bioinformatics")

        assert len(papers) == 1
        assert papers[0]["title"] == "A Bioinformatics Study"

    @patch("paperfind.fetchers.sources.biorxiv.requests.get")
    def test_fetch_biorxiv_category_normalization(self, mock_get):
        """Test that hyphens in category are normalized to spaces."""
        from paperfind.fetchers.sources.biorxiv import fetch_biorxiv

        response_data = {
            "collection": [
                {
                    "doi": "10.1101/test",
                    "title": "Test",
                    "authors": "Author",
                    "abstract": "Abstract",
                    "date": "2024-01-15",
                    "category": "systems biology",  # API uses spaces
                },
            ]
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        start_date = date(2024, 1, 14)
        end_date = date(2024, 1, 15)
        # Config uses hyphens
        papers = fetch_biorxiv(start_date, end_date, category="systems-biology")

        assert len(papers) == 1

    @patch("paperfind.fetchers.sources.biorxiv.requests.get")
    def test_fetch_biorxiv_empty_response(self, mock_get):
        """Test empty response handling."""
        from paperfind.fetchers.sources.biorxiv import fetch_biorxiv

        mock_response = MagicMock()
        mock_response.json.return_value = {"collection": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        papers = fetch_biorxiv(date(2024, 1, 1), date(2024, 1, 1))
        assert papers == []

    @patch("paperfind.fetchers.sources.biorxiv.requests.get")
    def test_fetch_biorxiv_network_error(self, mock_get):
        """Test network error handling."""
        from paperfind.fetchers.sources.biorxiv import fetch_biorxiv

        mock_get.side_effect = requests.RequestException("Connection failed")

        papers = fetch_biorxiv(date(2024, 1, 1), date(2024, 1, 1))
        assert papers == []

    @patch("paperfind.fetchers.sources.biorxiv.requests.get")
    def test_fetch_medrxiv_success(self, mock_get, sample_biorxiv_response):
        """Test fetching from medRxiv (same API, different server)."""
        from paperfind.fetchers.sources.biorxiv import fetch_biorxiv

        mock_response = MagicMock()
        mock_response.json.return_value = sample_biorxiv_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        papers = fetch_biorxiv(date(2024, 1, 14), date(2024, 1, 15), server="medrxiv")

        assert len(papers) == 2
        assert papers[0]["source"] == "medrxiv"

    @patch("paperfind.fetchers.sources.biorxiv.requests.get")
    def test_fetch_biorxiv_skips_missing_fields(self, mock_get):
        """Test that entries without required fields are skipped."""
        from paperfind.fetchers.sources.biorxiv import fetch_biorxiv

        response_data = {
            "collection": [
                {"doi": "10.1101/test1", "title": "", "abstract": "Has abstract"},
                {"doi": "10.1101/test2", "title": "Has title", "abstract": ""},
                {"doi": "", "title": "Has title", "abstract": "Has abstract"},
                {
                    "doi": "10.1101/valid",
                    "title": "Valid Paper",
                    "abstract": "Valid abstract",
                },
            ]
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        papers = fetch_biorxiv(date(2024, 1, 1), date(2024, 1, 1))
        assert len(papers) == 1
        assert papers[0]["doi"] == "10.1101/valid"


class TestCrossrefFetcher:
    """Tests for CrossRef paper fetcher."""

    @pytest.fixture
    def sample_crossref_response(self):
        """Sample CrossRef API response."""
        return {
            "message": {
                "items": [
                    {
                        "DOI": "10.1234/test.2024.001",
                        "title": ["A Research Paper"],
                        "author": [
                            {"given": "John", "family": "Doe"},
                            {"given": "Jane", "family": "Smith"},
                        ],
                        "abstract": "<p>This is the abstract.</p>",
                        "created": {"date-parts": [[2024, 1, 15]]},
                        "type": "journal-article",
                    },
                ],
                "next-cursor": None,
            }
        }

    @patch("paperfind.fetchers.sources.crossref.requests.get")
    def test_fetch_crossref_success(self, mock_get, sample_crossref_response):
        """Test successful fetch from CrossRef."""
        from paperfind.fetchers.sources.crossref import fetch_crossref

        mock_response = MagicMock()
        mock_response.json.return_value = sample_crossref_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        papers = fetch_crossref(date(2024, 1, 15))

        assert len(papers) == 1
        assert papers[0]["doi"] == "10.1234/test.2024.001"
        assert papers[0]["title"] == "A Research Paper"
        assert papers[0]["authors"] == "John Doe, Jane Smith"
        assert papers[0]["abstract"] == "This is the abstract."
        assert papers[0]["source"] == "crossref"
        assert papers[0]["type"] == "journal-article"

    @patch("paperfind.fetchers.sources.crossref.requests.get")
    def test_fetch_crossref_html_processing(self, mock_get):
        """Test that HTML tags are stripped and entities are decoded."""
        from paperfind.fetchers.sources.crossref import fetch_crossref

        # CrossRef API returns actual HTML tags (not escaped), with HTML entities
        response = {
            "message": {
                "items": [
                    {
                        "DOI": "10.1234/test",
                        "title": ["Test"],
                        "abstract": "<jats:p>Test &amp; abstract with <jats:italic>emphasis</jats:italic>.</jats:p>",
                        "created": {"date-parts": [[2024, 1, 15]]},
                    },
                ],
                "next-cursor": None,
            }
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        papers = fetch_crossref(date(2024, 1, 15))

        # Tags should be stripped, entities should be decoded
        assert "Test & abstract" in papers[0]["abstract"]
        assert "<jats:p>" not in papers[0]["abstract"]
        assert "<jats:italic>" not in papers[0]["abstract"]

    @patch("paperfind.fetchers.sources.crossref.requests.get")
    def test_fetch_crossref_empty_response(self, mock_get):
        """Test empty response handling."""
        from paperfind.fetchers.sources.crossref import fetch_crossref

        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"items": [], "next-cursor": None}}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        papers = fetch_crossref(date(2024, 1, 15))
        assert papers == []

    @patch("paperfind.fetchers.sources.crossref.requests.get")
    def test_fetch_crossref_network_error(self, mock_get):
        """Test network error handling."""
        from paperfind.fetchers.sources.crossref import fetch_crossref

        mock_get.side_effect = requests.RequestException("Connection failed")

        papers = fetch_crossref(date(2024, 1, 15))
        assert papers == []

    @patch("paperfind.fetchers.sources.crossref.requests.get")
    def test_fetch_crossref_pagination(self, mock_get):
        """Test cursor-based pagination."""
        from paperfind.fetchers.sources.crossref import fetch_crossref

        page1 = {
            "message": {
                "items": [
                    {
                        "DOI": "10.1234/paper1",
                        "title": ["Paper 1"],
                        "abstract": "Abstract 1",
                        "created": {"date-parts": [[2024, 1, 15]]},
                    },
                ],
                "next-cursor": "cursor123",
            }
        }
        page2 = {
            "message": {
                "items": [
                    {
                        "DOI": "10.1234/paper2",
                        "title": ["Paper 2"],
                        "abstract": "Abstract 2",
                        "created": {"date-parts": [[2024, 1, 15]]},
                    },
                ],
                "next-cursor": None,
            }
        }

        mock_response1 = MagicMock()
        mock_response1.json.return_value = page1
        mock_response1.raise_for_status = MagicMock()

        mock_response2 = MagicMock()
        mock_response2.json.return_value = page2
        mock_response2.raise_for_status = MagicMock()

        mock_get.side_effect = [mock_response1, mock_response2]

        papers = fetch_crossref(date(2024, 1, 15))

        assert len(papers) == 2
        assert papers[0]["doi"] == "10.1234/paper1"
        assert papers[1]["doi"] == "10.1234/paper2"

    @patch("paperfind.fetchers.sources.crossref.requests.get")
    def test_fetch_crossref_skips_missing_fields(self, mock_get):
        """Test that entries without required fields are skipped."""
        from paperfind.fetchers.sources.crossref import fetch_crossref

        response = {
            "message": {
                "items": [
                    {"DOI": "10.1234/no-title", "abstract": "Has abstract"},
                    {"DOI": "10.1234/no-abstract", "title": ["Has title"]},
                    {"title": ["No DOI"], "abstract": "Has abstract"},
                    {
                        "DOI": "10.1234/valid",
                        "title": ["Valid Paper"],
                        "abstract": "Valid abstract",
                        "created": {"date-parts": [[2024, 1, 15]]},
                    },
                ],
                "next-cursor": None,
            }
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        papers = fetch_crossref(date(2024, 1, 15))
        assert len(papers) == 1
        assert papers[0]["doi"] == "10.1234/valid"

    @patch("paperfind.fetchers.sources.crossref.requests.get")
    def test_fetch_crossref_with_type_filter(self, mock_get, sample_crossref_response):
        """Test type filtering parameter."""
        from paperfind.fetchers.sources.crossref import fetch_crossref

        mock_response = MagicMock()
        mock_response.json.return_value = sample_crossref_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        fetch_crossref(date(2024, 1, 15), type_filter="journal-article")

        # Verify the filter was included in the request
        call_args = mock_get.call_args
        assert "type:journal-article" in call_args[1]["params"]["filter"]


class TestChemrxivFetcher:
    """Tests for ChemRxiv paper fetcher."""

    @pytest.fixture
    def sample_chemrxiv_response(self):
        """Sample ChemRxiv API response."""
        return {
            "itemHits": [
                {
                    "item": {
                        "doi": "10.26434/chemrxiv-2024-abc123",
                        "title": "A Chemistry Paper",
                        "abstract": "This is a chemistry abstract.",
                        "authors": [
                            {"firstName": "John", "lastName": "Doe"},
                            {"firstName": "Jane", "lastName": "Smith"},
                        ],
                        "publishedDate": "2024-01-15T00:00:00Z",
                    }
                },
                {
                    "item": {
                        "doi": "10.26434/chemrxiv-2024-def456",
                        "title": "Another Chemistry Paper",
                        "abstract": "Another chemistry abstract.",
                        "authors": [{"lastName": "Wilson"}],
                        "publishedDate": "2024-01-14T00:00:00Z",
                    }
                },
            ]
        }

    @patch("paperfind.fetchers.sources.chemrxiv.requests.get")
    def test_fetch_chemrxiv_success(self, mock_get, sample_chemrxiv_response):
        """Test successful fetch from ChemRxiv."""
        from paperfind.fetchers.sources.chemrxiv import fetch_chemrxiv

        mock_response = MagicMock()
        mock_response.json.return_value = sample_chemrxiv_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        start_date = date(2024, 1, 14)
        end_date = date(2024, 1, 15)
        papers = fetch_chemrxiv(start_date, end_date)

        assert len(papers) == 2
        assert papers[0]["doi"] == "10.26434/chemrxiv-2024-abc123"
        assert papers[0]["title"] == "A Chemistry Paper"
        assert papers[0]["authors"] == "John Doe, Jane Smith"
        assert papers[0]["source"] == "chemrxiv"
        assert papers[0]["type"] == "preprint"

    @patch("paperfind.fetchers.sources.chemrxiv.requests.get")
    def test_fetch_chemrxiv_author_formatting(self, mock_get):
        """Test various author name formats."""
        from paperfind.fetchers.sources.chemrxiv import fetch_chemrxiv

        response = {
            "itemHits": [
                {
                    "item": {
                        "doi": "10.26434/test",
                        "title": "Test",
                        "abstract": "Abstract",
                        "authors": [
                            {"firstName": "John", "lastName": "Doe"},
                            {"lastName": "Smith"},  # No first name
                            {"firstName": "", "lastName": "Wilson"},  # Empty first name
                        ],
                        "publishedDate": "2024-01-15T00:00:00Z",
                    }
                },
            ]
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        papers = fetch_chemrxiv(date(2024, 1, 15), date(2024, 1, 15))

        assert papers[0]["authors"] == "John Doe, Smith, Wilson"

    @patch("paperfind.fetchers.sources.chemrxiv.requests.get")
    def test_fetch_chemrxiv_empty_response(self, mock_get):
        """Test empty response handling."""
        from paperfind.fetchers.sources.chemrxiv import fetch_chemrxiv

        mock_response = MagicMock()
        mock_response.json.return_value = {"itemHits": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        papers = fetch_chemrxiv(date(2024, 1, 15), date(2024, 1, 15))
        assert papers == []

    @patch("paperfind.fetchers.sources.chemrxiv.requests.get")
    def test_fetch_chemrxiv_network_error(self, mock_get):
        """Test network error handling."""
        from paperfind.fetchers.sources.chemrxiv import fetch_chemrxiv

        mock_get.side_effect = requests.RequestException("Connection failed")

        papers = fetch_chemrxiv(date(2024, 1, 15), date(2024, 1, 15))
        assert papers == []

    @patch("paperfind.fetchers.sources.chemrxiv.requests.get")
    def test_fetch_chemrxiv_date_filtering(self, mock_get):
        """Test that papers outside date range are filtered."""
        from paperfind.fetchers.sources.chemrxiv import fetch_chemrxiv

        response = {
            "itemHits": [
                {
                    "item": {
                        "doi": "10.26434/in-range",
                        "title": "In Range",
                        "abstract": "Abstract",
                        "authors": [],
                        "publishedDate": "2024-01-15T00:00:00Z",
                    }
                },
                {
                    "item": {
                        "doi": "10.26434/before-range",
                        "title": "Before Range",
                        "abstract": "Abstract",
                        "authors": [],
                        "publishedDate": "2024-01-10T00:00:00Z",
                    }
                },
                {
                    "item": {
                        "doi": "10.26434/after-range",
                        "title": "After Range",
                        "abstract": "Abstract",
                        "authors": [],
                        "publishedDate": "2024-01-20T00:00:00Z",
                    }
                },
            ]
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        papers = fetch_chemrxiv(date(2024, 1, 14), date(2024, 1, 16))

        assert len(papers) == 1
        assert papers[0]["doi"] == "10.26434/in-range"

    @patch("paperfind.fetchers.sources.chemrxiv.requests.get")
    def test_fetch_chemrxiv_skips_missing_fields(self, mock_get):
        """Test that entries without required fields are skipped."""
        from paperfind.fetchers.sources.chemrxiv import fetch_chemrxiv

        response = {
            "itemHits": [
                {"item": {"doi": "10.26434/no-title", "abstract": "Has abstract", "publishedDate": "2024-01-15"}},
                {"item": {"doi": "10.26434/no-abstract", "title": "Has title", "publishedDate": "2024-01-15"}},
                {"item": {"title": "No DOI", "abstract": "Has abstract", "publishedDate": "2024-01-15"}},
                {"item": {"doi": "10.26434/no-date", "title": "Has title", "abstract": "Has abstract"}},
                {
                    "item": {
                        "doi": "10.26434/valid",
                        "title": "Valid Paper",
                        "abstract": "Valid abstract",
                        "authors": [],
                        "publishedDate": "2024-01-15T00:00:00Z",
                    }
                },
            ]
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        papers = fetch_chemrxiv(date(2024, 1, 14), date(2024, 1, 16))
        assert len(papers) == 1
        assert papers[0]["doi"] == "10.26434/valid"

    @patch("paperfind.fetchers.sources.chemrxiv.requests.get")
    def test_fetch_chemrxiv_pagination(self, mock_get):
        """Test pagination through multiple pages."""
        from paperfind.fetchers.sources.chemrxiv import fetch_chemrxiv

        # First page with 50 items (full page)
        page1_items = [
            {
                "item": {
                    "doi": f"10.26434/paper{i}",
                    "title": f"Paper {i}",
                    "abstract": "Abstract",
                    "authors": [],
                    "publishedDate": "2024-01-15T00:00:00Z",
                }
            }
            for i in range(50)
        ]
        # Second page with fewer items (last page)
        page2_items = [
            {
                "item": {
                    "doi": "10.26434/paper50",
                    "title": "Paper 50",
                    "abstract": "Abstract",
                    "authors": [],
                    "publishedDate": "2024-01-15T00:00:00Z",
                }
            }
        ]

        mock_response1 = MagicMock()
        mock_response1.json.return_value = {"itemHits": page1_items}
        mock_response1.raise_for_status = MagicMock()

        mock_response2 = MagicMock()
        mock_response2.json.return_value = {"itemHits": page2_items}
        mock_response2.raise_for_status = MagicMock()

        mock_get.side_effect = [mock_response1, mock_response2]

        papers = fetch_chemrxiv(date(2024, 1, 15), date(2024, 1, 15))

        assert len(papers) == 51


class TestExceptions:
    """Tests for custom exception classes."""

    def test_fetcher_error_attributes(self):
        """Test FetcherError has correct attributes."""
        from paperfind.exceptions import FetcherError

        error = FetcherError("arxiv", "Connection failed")
        assert error.source == "arxiv"
        assert error.message == "Connection failed"
        assert str(error) == "[arxiv] Connection failed"

    def test_exception_hierarchy(self):
        """Test exception class hierarchy."""
        from paperfind.exceptions import (
            ConfigError,
            EmbeddingError,
            FetcherError,
            PaperfindError,
            VectorStoreError,
            ZoteroError,
        )

        assert issubclass(ConfigError, PaperfindError)
        assert issubclass(FetcherError, PaperfindError)
        assert issubclass(VectorStoreError, PaperfindError)
        assert issubclass(EmbeddingError, PaperfindError)
        assert issubclass(ZoteroError, PaperfindError)
        assert issubclass(PaperfindError, Exception)
