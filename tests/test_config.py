"""Tests for configuration validation."""

from unittest.mock import patch

import pytest

from paperfind.config import (
    ConfigValidationError,
    check_config,
    get_config_status,
    validate_config,
)


class TestValidateConfig:
    """Tests for validate_config function."""

    @patch.dict("os.environ", {}, clear=True)
    @patch("paperfind.config.ZOTERO_API_KEY", None)
    @patch("paperfind.config.ZOTERO_USER_ID", None)
    def test_zotero_missing_both(self):
        """Test zotero validation with both keys missing."""
        missing = validate_config("zotero", raise_on_error=False)
        assert "ZOTERO_API_KEY" in missing
        assert "ZOTERO_USER_ID" in missing

    @patch("paperfind.config.ZOTERO_API_KEY", "test_key")
    @patch("paperfind.config.ZOTERO_USER_ID", "test_id")
    def test_zotero_all_present(self):
        """Test zotero validation with all keys present."""
        missing = validate_config("zotero", raise_on_error=False)
        assert missing == []

    @patch("paperfind.config.ZOTERO_API_KEY", None)
    @patch("paperfind.config.ZOTERO_USER_ID", None)
    def test_zotero_raises_on_error(self):
        """Test that validation raises ConfigValidationError when configured."""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_config("zotero", raise_on_error=True)
        assert "ZOTERO_API_KEY" in exc_info.value.missing
        assert exc_info.value.context == "zotero"

    @patch("paperfind.embeddings.get_embedding_provider", return_value="openai")
    @patch("paperfind.config.OPENAI_API_KEY", None)
    def test_embeddings_openai_missing_key(self, mock_provider):
        """Test embeddings validation for OpenAI without API key."""
        missing = validate_config("embeddings", raise_on_error=False)
        assert "OPENAI_API_KEY" in missing

    @patch("paperfind.embeddings.get_embedding_provider", return_value="openai")
    @patch("paperfind.config.OPENAI_API_KEY", "test_key")
    def test_embeddings_openai_with_key(self, mock_provider):
        """Test embeddings validation for OpenAI with API key."""
        missing = validate_config("embeddings", raise_on_error=False)
        assert missing == []

    @patch("paperfind.embeddings.get_embedding_provider", return_value="ollama")
    def test_embeddings_ollama_no_key_needed(self, mock_provider):
        """Test that Ollama doesn't require an API key."""
        missing = validate_config("embeddings", raise_on_error=False)
        assert missing == []

    @patch("paperfind.embeddings.get_embedding_provider", return_value="huggingface")
    def test_embeddings_huggingface_no_key_needed(self, mock_provider):
        """Test that HuggingFace doesn't require an API key."""
        missing = validate_config("embeddings", raise_on_error=False)
        assert missing == []

    @patch("paperfind.config.SMTP_USER", None)
    @patch("paperfind.config.SMTP_PASSWORD", None)
    @patch("paperfind.config.EMAIL_FROM", None)
    @patch("paperfind.config.EMAIL_TO", None)
    def test_email_all_missing(self):
        """Test email validation with all settings missing."""
        missing = validate_config("email", raise_on_error=False)
        assert "SMTP_USER" in missing
        assert "SMTP_PASSWORD" in missing
        assert "EMAIL_FROM" in missing
        assert "EMAIL_TO" in missing

    @patch("paperfind.config.SMTP_USER", "user")
    @patch("paperfind.config.SMTP_PASSWORD", "pass")
    @patch("paperfind.config.EMAIL_FROM", "from@test.com")
    @patch("paperfind.config.EMAIL_TO", "to@test.com")
    def test_email_all_present(self):
        """Test email validation with all settings present."""
        missing = validate_config("email", raise_on_error=False)
        assert missing == []

    def test_general_validation_no_requirements(self):
        """Test general validation has no specific requirements."""
        missing = validate_config("general", raise_on_error=False)
        assert missing == []


class TestCheckConfig:
    """Tests for check_config convenience function."""

    @patch("paperfind.config.ZOTERO_API_KEY", "key")
    @patch("paperfind.config.ZOTERO_USER_ID", "id")
    def test_returns_true_when_valid(self):
        """Test check_config returns True when config is valid."""
        assert check_config("zotero") is True

    @patch("paperfind.config.ZOTERO_API_KEY", None)
    @patch("paperfind.config.ZOTERO_USER_ID", None)
    def test_returns_false_when_invalid(self):
        """Test check_config returns False when config is invalid."""
        assert check_config("zotero") is False


class TestConfigValidationError:
    """Tests for ConfigValidationError exception."""

    def test_error_message_format(self):
        """Test error message includes missing variables."""
        error = ConfigValidationError(["VAR1", "VAR2"], "test")
        assert "VAR1" in str(error)
        assert "VAR2" in str(error)
        assert "test" in str(error)

    def test_error_attributes(self):
        """Test error has correct attributes."""
        error = ConfigValidationError(["VAR1"], "context")
        assert error.missing == ["VAR1"]
        assert error.context == "context"

    def test_error_without_context(self):
        """Test error message without context."""
        error = ConfigValidationError(["VAR1"])
        assert "VAR1" in str(error)


class TestGetConfigStatus:
    """Tests for get_config_status function."""

    @patch("paperfind.embeddings.get_embedding_provider", return_value="openai")
    @patch("paperfind.embeddings.get_embedding_model", return_value="text-embedding-3-small")
    @patch("paperfind.config.ZOTERO_API_KEY", "key")
    @patch("paperfind.config.ZOTERO_USER_ID", "id")
    @patch("paperfind.config.OPENAI_API_KEY", "key")
    def test_returns_status_dict(self, mock_model, mock_provider):
        """Test get_config_status returns proper structure."""
        status = get_config_status()

        assert "data_dir" in status
        assert "env_file_loaded" in status
        assert "embedding_provider" in status
        assert "embedding_model" in status
        assert "operations" in status
        assert "zotero" in status["operations"]
        assert "embeddings" in status["operations"]
        assert "email" in status["operations"]
