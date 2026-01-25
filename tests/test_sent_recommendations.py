"""Tests for sent recommendations tracking."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch


class TestSentRecommendations:
    """Tests for sent_recommendations database functions."""

    @patch("paperfind.fetchers.db.get_db")
    @patch("paperfind.fetchers.db.table_exists")
    def test_get_sent_dois_returns_empty_when_no_table(
        self, mock_table_exists, mock_get_db
    ):
        from paperfind.fetchers.db import get_sent_dois

        mock_conn = MagicMock()
        mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_db.return_value.__exit__ = MagicMock(return_value=False)
        mock_table_exists.return_value = False

        result = get_sent_dois()

        assert result == set()

    @patch("paperfind.fetchers.db.get_db")
    @patch("paperfind.fetchers.db.table_exists")
    @patch("paperfind.fetchers.db.qualify_table")
    def test_get_sent_dois_returns_dois(
        self, mock_qualify, mock_table_exists, mock_get_db
    ):
        from paperfind.fetchers.db import get_sent_dois

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_db.return_value.__exit__ = MagicMock(return_value=False)
        mock_table_exists.return_value = True
        mock_qualify.return_value = "daily.sent_recommendations"

        mock_cursor.fetchall.return_value = [
            {"doi": "10.1234/a"},
            {"doi": "10.1234/b"},
        ]

        result = get_sent_dois()

        assert result == {"10.1234/a", "10.1234/b"}

    @patch("paperfind.fetchers.db.init_sent_recommendations_table")
    @patch("paperfind.fetchers.db.get_db")
    @patch("paperfind.fetchers.db.qualify_table")
    @patch("paperfind.fetchers.db.placeholders")
    def test_record_sent_dois_inserts_dois(
        self, mock_placeholders, mock_qualify, mock_get_db, mock_init
    ):
        from paperfind.fetchers.db import record_sent_dois

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_db.return_value.__exit__ = MagicMock(return_value=False)
        mock_qualify.return_value = "daily.sent_recommendations"
        mock_placeholders.return_value = "?, ?"

        dois = ["10.1234/a", "10.1234/b", "10.1234/c"]
        result = record_sent_dois(dois)

        assert result == 3
        assert mock_cursor.execute.call_count == 3
        mock_conn.commit.assert_called_once()

    @patch("paperfind.fetchers.db.init_sent_recommendations_table")
    @patch("paperfind.fetchers.db.get_db")
    def test_record_sent_dois_empty_list(self, mock_get_db, mock_init):
        from paperfind.fetchers.db import record_sent_dois

        result = record_sent_dois([])

        assert result == 0
        mock_get_db.assert_not_called()

    @patch("paperfind.fetchers.db.get_db")
    @patch("paperfind.fetchers.db.table_exists")
    def test_prune_sent_recommendations_no_table(
        self, mock_table_exists, mock_get_db
    ):
        from paperfind.fetchers.db import prune_sent_recommendations

        mock_conn = MagicMock()
        mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_db.return_value.__exit__ = MagicMock(return_value=False)
        mock_table_exists.return_value = False

        result = prune_sent_recommendations(date.today())

        assert result == 0

    @patch("paperfind.fetchers.db.get_db")
    @patch("paperfind.fetchers.db.table_exists")
    @patch("paperfind.fetchers.db.qualify_table")
    @patch("paperfind.fetchers.db.placeholder")
    def test_prune_sent_recommendations_deletes_old(
        self, mock_placeholder, mock_qualify, mock_table_exists, mock_get_db
    ):
        from paperfind.fetchers.db import prune_sent_recommendations

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 5
        mock_conn.cursor.return_value = mock_cursor
        mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_db.return_value.__exit__ = MagicMock(return_value=False)
        mock_table_exists.return_value = True
        mock_qualify.return_value = "daily.sent_recommendations"
        mock_placeholder.return_value = "?"

        cutoff = date.today() - timedelta(days=30)
        result = prune_sent_recommendations(cutoff)

        assert result == 5
        mock_conn.commit.assert_called_once()

    @patch("paperfind.fetchers.db.get_db")
    @patch("paperfind.fetchers.db.qualify_table")
    def test_init_sent_recommendations_table_creates_table(
        self, mock_qualify, mock_get_db
    ):
        from paperfind.fetchers.db import init_sent_recommendations_table

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_get_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_db.return_value.__exit__ = MagicMock(return_value=False)
        mock_qualify.return_value = "daily.sent_recommendations"

        init_sent_recommendations_table()

        mock_cursor.execute.assert_called_once()
        mock_conn.commit.assert_called_once()
