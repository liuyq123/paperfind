"""Integration tests for the digest workflow."""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


class TestDigestWorkflowDryRun:
    """Tests for digest workflow in dry-run mode."""

    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.render_digest")
    @patch("paperfind.digest.digest.get_sent_dois")
    def test_dry_run_skips_email_and_recording(
        self,
        mock_get_sent_dois,
        mock_render,
        mock_get_recs,
    ):
        """Dry run should render HTML but not send email or record DOIs."""
        from paperfind.digest.digest import run_digest

        doc = Document(
            page_content="Test Paper\n\nAbstract here.",
            metadata={"doi": "10.1234/test", "title": "Test Paper"},
        )
        mock_get_recs.return_value = ([("10.1234/test", (0.9, doc, "Zotero Paper"))], True)
        mock_render.return_value = "<html>Test</html>"
        mock_get_sent_dois.return_value = set()

        # Should not raise, should not call email or record functions
        with patch("paperfind.digest.digest.send_email") as mock_send:
            with patch("paperfind.digest.digest.record_sent_dois") as mock_record:
                run_digest(dry_run=True, skip_fetch=True)

                mock_send.assert_not_called()
                mock_record.assert_not_called()
                mock_render.assert_called_once()

    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.get_sent_dois")
    def test_dry_run_with_no_recommendations(
        self,
        mock_get_sent_dois,
        mock_get_recs,
    ):
        """Dry run should handle no recommendations gracefully."""
        from paperfind.digest.digest import run_digest

        mock_get_recs.return_value = ([], False)
        mock_get_sent_dois.return_value = set()

        with patch("paperfind.digest.digest.render_digest") as mock_render:
            run_digest(dry_run=True, skip_fetch=True)

            # Should not render if no recommendations
            mock_render.assert_not_called()


class TestDigestWorkflowWithEmail:
    """Tests for digest workflow with email sending."""

    @patch("paperfind.digest.digest.EMAIL_TO", "test@example.com")
    @patch("paperfind.digest.digest.EMAIL_FROM", "from@example.com")
    @patch("paperfind.digest.digest.SMTP_USER", "user")
    @patch("paperfind.digest.digest.SMTP_PASSWORD", "pass")
    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.render_digest")
    @patch("paperfind.digest.digest.send_email")
    @patch("paperfind.digest.digest.get_sent_dois")
    @patch("paperfind.digest.digest.record_sent_dois")
    @patch("paperfind.digest.digest.prune_sent_recommendations")
    def test_full_workflow_sends_email_and_records(
        self,
        mock_prune,
        mock_record,
        mock_get_sent_dois,
        mock_send,
        mock_render,
        mock_get_recs,
    ):
        """Full workflow should send email, record DOIs, and prune old records."""
        from paperfind.digest.digest import run_digest

        doc = Document(
            page_content="Test Paper\n\nAbstract here.",
            metadata={"doi": "10.1234/test", "title": "Test Paper"},
        )
        mock_get_recs.return_value = ([("10.1234/test", (0.9, doc, "Zotero Paper"))], True)
        mock_render.return_value = "<html>Test</html>"
        mock_get_sent_dois.return_value = set()

        run_digest(dry_run=False, skip_fetch=True)

        mock_send.assert_called_once()
        mock_record.assert_called_once_with(["10.1234/test"])
        mock_prune.assert_called_once()

    @patch("paperfind.digest.digest.EMAIL_TO", "test@example.com")
    @patch("paperfind.digest.digest.EMAIL_FROM", "from@example.com")
    @patch("paperfind.digest.digest.SMTP_USER", "user")
    @patch("paperfind.digest.digest.SMTP_PASSWORD", "pass")
    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.render_digest")
    @patch("paperfind.digest.digest.send_email")
    @patch("paperfind.digest.digest.get_sent_dois")
    @patch("paperfind.digest.digest.record_sent_dois")
    @patch("paperfind.digest.digest.prune_sent_recommendations")
    def test_multiple_recommendations_all_recorded(
        self,
        mock_prune,
        mock_record,
        mock_get_sent_dois,
        mock_send,
        mock_render,
        mock_get_recs,
    ):
        """All recommended DOIs should be recorded after email send."""
        from paperfind.digest.digest import run_digest

        doc1 = Document(page_content="Paper 1", metadata={"doi": "10.1234/a"})
        doc2 = Document(page_content="Paper 2", metadata={"doi": "10.1234/b"})
        doc3 = Document(page_content="Paper 3", metadata={"doi": "10.1234/c"})

        mock_get_recs.return_value = (
            [
                ("10.1234/a", (0.9, doc1, "Seed 1")),
                ("10.1234/b", (0.8, doc2, "Seed 2")),
                ("10.1234/c", (0.7, doc3, "Seed 3")),
            ],
            True,
        )
        mock_render.return_value = "<html>Test</html>"
        mock_get_sent_dois.return_value = set()

        run_digest(dry_run=False, skip_fetch=True)

        mock_record.assert_called_once_with(["10.1234/a", "10.1234/b", "10.1234/c"])

    @patch("paperfind.digest.digest.EMAIL_TO", "test@example.com")
    @patch("paperfind.digest.digest.EMAIL_FROM", "from@example.com")
    @patch("paperfind.digest.digest.SMTP_USER", "user")
    @patch("paperfind.digest.digest.SMTP_PASSWORD", "pass")
    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.render_digest")
    @patch("paperfind.digest.digest.send_email")
    @patch("paperfind.digest.digest.get_sent_dois")
    @patch("paperfind.digest.digest.record_sent_dois")
    def test_email_failure_does_not_record_dois(
        self,
        mock_record,
        mock_get_sent_dois,
        mock_send,
        mock_render,
        mock_get_recs,
    ):
        """If email sending fails, DOIs should not be recorded."""
        from paperfind.digest.digest import run_digest

        doc = Document(page_content="Test Paper", metadata={"doi": "10.1234/test"})
        mock_get_recs.return_value = ([("10.1234/test", (0.9, doc, "Seed"))], True)
        mock_render.return_value = "<html>Test</html>"
        mock_get_sent_dois.return_value = set()
        mock_send.side_effect = ValueError("SMTP error")

        run_digest(dry_run=False, skip_fetch=True)

        mock_record.assert_not_called()


class TestSentDoisExclusion:
    """Tests for excluding previously sent DOIs."""

    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.get_sent_dois")
    def test_sent_dois_passed_to_recommendations(
        self,
        mock_get_sent_dois,
        mock_get_recs,
    ):
        """Previously sent DOIs should be passed to get_recommendations."""
        from paperfind.digest.digest import run_digest

        mock_get_sent_dois.return_value = {"10.1234/old1", "10.1234/old2"}
        mock_get_recs.return_value = ([], False)

        run_digest(dry_run=True, skip_fetch=True)

        # Verify exclude_dois was passed
        call_kwargs = mock_get_recs.call_args[1]
        assert call_kwargs["exclude_dois"] == {"10.1234/old1", "10.1234/old2"}

    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.get_sent_dois")
    def test_empty_sent_dois_passed_as_empty_set(
        self,
        mock_get_sent_dois,
        mock_get_recs,
    ):
        """Empty sent DOIs should be passed as empty set."""
        from paperfind.digest.digest import run_digest

        mock_get_sent_dois.return_value = set()
        mock_get_recs.return_value = ([], False)

        run_digest(dry_run=True, skip_fetch=True)

        call_kwargs = mock_get_recs.call_args[1]
        assert call_kwargs["exclude_dois"] == set()


class TestDigestConfigValidation:
    """Tests for configuration validation in digest."""

    @patch("paperfind.digest.digest.EMAIL_TO", "")
    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.render_digest")
    @patch("paperfind.digest.digest.get_sent_dois")
    @patch("paperfind.digest.digest.send_email")
    def test_missing_email_to_does_not_send(
        self,
        mock_send,
        mock_get_sent_dois,
        mock_render,
        mock_get_recs,
    ):
        """Missing EMAIL_TO should prevent sending."""
        from paperfind.digest.digest import run_digest

        doc = Document(page_content="Test", metadata={"doi": "10.1234/test"})
        mock_get_recs.return_value = ([("10.1234/test", (0.9, doc, "Seed"))], True)
        mock_render.return_value = "<html>Test</html>"
        mock_get_sent_dois.return_value = set()

        run_digest(dry_run=False, skip_fetch=True)

        mock_send.assert_not_called()

    @patch("paperfind.digest.digest.EMAIL_TO", "test@example.com")
    @patch("paperfind.digest.digest.EMAIL_FROM", "")
    @patch("paperfind.digest.digest.SMTP_USER", "user")
    @patch("paperfind.digest.digest.SMTP_PASSWORD", "pass")
    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.render_digest")
    @patch("paperfind.digest.digest.get_sent_dois")
    @patch("paperfind.digest.digest.send_email")
    def test_missing_email_from_does_not_send(
        self,
        mock_send,
        mock_get_sent_dois,
        mock_render,
        mock_get_recs,
    ):
        """Missing EMAIL_FROM should prevent sending."""
        from paperfind.digest.digest import run_digest

        doc = Document(page_content="Test", metadata={"doi": "10.1234/test"})
        mock_get_recs.return_value = ([("10.1234/test", (0.9, doc, "Seed"))], True)
        mock_render.return_value = "<html>Test</html>"
        mock_get_sent_dois.return_value = set()

        run_digest(dry_run=False, skip_fetch=True)

        mock_send.assert_not_called()

    @patch("paperfind.digest.digest.EMAIL_TO", "test@example.com")
    @patch("paperfind.digest.digest.EMAIL_FROM", "from@example.com")
    @patch("paperfind.digest.digest.SMTP_USER", "")
    @patch("paperfind.digest.digest.SMTP_PASSWORD", "")
    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.render_digest")
    @patch("paperfind.digest.digest.get_sent_dois")
    @patch("paperfind.digest.digest.send_email")
    def test_missing_smtp_credentials_does_not_send(
        self,
        mock_send,
        mock_get_sent_dois,
        mock_render,
        mock_get_recs,
    ):
        """Missing SMTP credentials should prevent sending."""
        from paperfind.digest.digest import run_digest

        doc = Document(page_content="Test", metadata={"doi": "10.1234/test"})
        mock_get_recs.return_value = ([("10.1234/test", (0.9, doc, "Seed"))], True)
        mock_render.return_value = "<html>Test</html>"
        mock_get_sent_dois.return_value = set()

        run_digest(dry_run=False, skip_fetch=True)

        mock_send.assert_not_called()


class TestDigestWithFetch:
    """Tests for digest workflow with paper fetching."""

    @patch("paperfind.digest.digest.fetch_all")
    @patch("paperfind.digest.digest.upsert_vectors_for_dois")
    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.get_sent_dois")
    def test_fetch_called_when_not_skipped(
        self,
        mock_get_sent_dois,
        mock_get_recs,
        mock_upsert,
        mock_fetch,
    ):
        """Fetch should be called when skip_fetch is False."""
        from paperfind.digest.digest import run_digest

        mock_fetch.return_value = ({"arxiv": 5, "crossref": 3}, ["10.1234/a", "10.1234/b"])
        mock_get_recs.return_value = ([], False)
        mock_get_sent_dois.return_value = set()

        run_digest(days=3, dry_run=True, skip_fetch=False)

        mock_fetch.assert_called_once()
        assert mock_fetch.call_args[1]["days"] == 3
        mock_upsert.assert_called_once()

    @patch("paperfind.digest.digest.fetch_all")
    @patch("paperfind.digest.digest.upsert_vectors_for_dois")
    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.get_sent_dois")
    def test_fetch_skipped_when_requested(
        self,
        mock_get_sent_dois,
        mock_get_recs,
        mock_upsert,
        mock_fetch,
    ):
        """Fetch should not be called when skip_fetch is True."""
        from paperfind.digest.digest import run_digest

        mock_get_recs.return_value = ([], False)
        mock_get_sent_dois.return_value = set()

        run_digest(dry_run=True, skip_fetch=True)

        mock_fetch.assert_not_called()
        mock_upsert.assert_not_called()

    @patch("paperfind.digest.digest.fetch_all")
    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.get_sent_dois")
    def test_invalid_days_returns_early(
        self,
        mock_get_sent_dois,
        mock_get_recs,
        mock_fetch,
    ):
        """Invalid days parameter should return early without fetching."""
        from paperfind.digest.digest import run_digest

        run_digest(days=0, dry_run=True, skip_fetch=False)

        mock_fetch.assert_not_called()
        mock_get_recs.assert_not_called()

    @patch("paperfind.digest.digest.fetch_all")
    @patch("paperfind.digest.digest.upsert_vectors_for_dois")
    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.get_sent_dois")
    def test_arxiv_days_passed_to_fetch(
        self,
        mock_get_sent_dois,
        mock_get_recs,
        mock_upsert,
        mock_fetch,
    ):
        """arxiv_days parameter should be passed to fetch_all."""
        from paperfind.digest.digest import run_digest

        mock_fetch.return_value = ({}, [])
        mock_get_recs.return_value = ([], False)
        mock_get_sent_dois.return_value = set()

        run_digest(days=1, arxiv_days=3, dry_run=True, skip_fetch=False)

        assert mock_fetch.call_args[1]["arxiv_days"] == 3


class TestDigestRecommendationParameters:
    """Tests for recommendation parameter passing."""

    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.get_sent_dois")
    def test_num_recommendations_passed(
        self,
        mock_get_sent_dois,
        mock_get_recs,
    ):
        """num_recommendations should be passed as k."""
        from paperfind.digest.digest import run_digest

        mock_get_recs.return_value = ([], False)
        mock_get_sent_dois.return_value = set()

        run_digest(num_recommendations=15, dry_run=True, skip_fetch=True)

        assert mock_get_recs.call_args[1]["k"] == 15

    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.get_sent_dois")
    def test_collection_passed(
        self,
        mock_get_sent_dois,
        mock_get_recs,
    ):
        """collection should be passed to get_recommendations."""
        from paperfind.digest.digest import run_digest

        mock_get_recs.return_value = ([], False)
        mock_get_sent_dois.return_value = set()

        run_digest(collection="My Collection", dry_run=True, skip_fetch=True)

        assert mock_get_recs.call_args[1]["collection"] == "My Collection"

    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.get_sent_dois")
    def test_rerank_passed(
        self,
        mock_get_sent_dois,
        mock_get_recs,
    ):
        """rerank should be passed to get_recommendations."""
        from paperfind.digest.digest import run_digest

        mock_get_recs.return_value = ([], False)
        mock_get_sent_dois.return_value = set()

        run_digest(rerank=False, dry_run=True, skip_fetch=True)

        assert mock_get_recs.call_args[1]["rerank"] is False

    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.get_sent_dois")
    def test_max_age_days_passed(
        self,
        mock_get_sent_dois,
        mock_get_recs,
    ):
        """max_age_days should be passed to get_recommendations."""
        from paperfind.digest.digest import run_digest

        mock_get_recs.return_value = ([], False)
        mock_get_sent_dois.return_value = set()

        run_digest(max_age_days=7, dry_run=True, skip_fetch=True)

        assert mock_get_recs.call_args[1]["max_age_days"] == 7


class TestDigestTemplateRendering:
    """Tests for template rendering in digest workflow."""

    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.render_digest")
    @patch("paperfind.digest.digest.get_sent_dois")
    def test_render_called_with_recommendations(
        self,
        mock_get_sent_dois,
        mock_render,
        mock_get_recs,
    ):
        """render_digest should be called with recommendations and date."""
        from paperfind.digest.digest import run_digest

        doc = Document(page_content="Test", metadata={"doi": "10.1234/test"})
        recommendations = [("10.1234/test", (0.9, doc, "Seed"))]
        mock_get_recs.return_value = (recommendations, True)
        mock_render.return_value = "<html>Test</html>"
        mock_get_sent_dois.return_value = set()

        run_digest(dry_run=True, skip_fetch=True)

        mock_render.assert_called_once()
        call_args = mock_render.call_args[0]
        assert call_args[0] == recommendations
        assert isinstance(call_args[1], date)

    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.render_digest")
    @patch("paperfind.digest.digest.get_sent_dois")
    def test_render_receives_rerank_flag(
        self,
        mock_get_sent_dois,
        mock_render,
        mock_get_recs,
    ):
        """render_digest should receive the rerank_used flag."""
        from paperfind.digest.digest import run_digest

        doc = Document(page_content="Test", metadata={"doi": "10.1234/test"})
        mock_get_recs.return_value = ([("10.1234/test", (0.9, doc, "Seed"))], False)
        mock_render.return_value = "<html>Test</html>"
        mock_get_sent_dois.return_value = set()

        run_digest(dry_run=True, skip_fetch=True)

        assert mock_render.call_args[1]["rerank"] is False


class TestDigestPruning:
    """Tests for pruning old sent recommendations."""

    @patch("paperfind.digest.digest.EMAIL_TO", "test@example.com")
    @patch("paperfind.digest.digest.EMAIL_FROM", "from@example.com")
    @patch("paperfind.digest.digest.SMTP_USER", "user")
    @patch("paperfind.digest.digest.SMTP_PASSWORD", "pass")
    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.render_digest")
    @patch("paperfind.digest.digest.send_email")
    @patch("paperfind.digest.digest.get_sent_dois")
    @patch("paperfind.digest.digest.record_sent_dois")
    @patch("paperfind.digest.digest.prune_sent_recommendations")
    def test_prune_called_with_30_day_cutoff(
        self,
        mock_prune,
        mock_record,
        mock_get_sent_dois,
        mock_send,
        mock_render,
        mock_get_recs,
    ):
        """Prune should be called with 30-day cutoff after successful email."""
        from paperfind.digest.digest import run_digest

        doc = Document(page_content="Test", metadata={"doi": "10.1234/test"})
        mock_get_recs.return_value = ([("10.1234/test", (0.9, doc, "Seed"))], True)
        mock_render.return_value = "<html>Test</html>"
        mock_get_sent_dois.return_value = set()

        run_digest(dry_run=False, skip_fetch=True)

        mock_prune.assert_called_once()
        cutoff = mock_prune.call_args[0][0]
        expected_cutoff = date.today() - timedelta(days=30)
        assert cutoff == expected_cutoff

    @patch("paperfind.digest.digest.get_recommendations")
    @patch("paperfind.digest.digest.render_digest")
    @patch("paperfind.digest.digest.get_sent_dois")
    @patch("paperfind.digest.digest.prune_sent_recommendations")
    def test_prune_not_called_in_dry_run(
        self,
        mock_prune,
        mock_get_sent_dois,
        mock_render,
        mock_get_recs,
    ):
        """Prune should not be called in dry-run mode."""
        from paperfind.digest.digest import run_digest

        doc = Document(page_content="Test", metadata={"doi": "10.1234/test"})
        mock_get_recs.return_value = ([("10.1234/test", (0.9, doc, "Seed"))], True)
        mock_render.return_value = "<html>Test</html>"
        mock_get_sent_dois.return_value = set()

        run_digest(dry_run=True, skip_fetch=True)

        mock_prune.assert_not_called()
