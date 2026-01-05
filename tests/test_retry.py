"""Tests for retry utility."""

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from paperfind.retry import (
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_RETRIES,
    calculate_delay,
    is_retryable_status,
    retry_request,
    with_retry,
)


class TestIsRetryableStatus:
    """Tests for is_retryable_status."""

    def test_retryable_status_codes(self):
        """Test that retryable status codes return True."""
        retryable = [429, 500, 502, 503, 504]
        for code in retryable:
            assert is_retryable_status(code) is True, f"{code} should be retryable"

    def test_non_retryable_status_codes(self):
        """Test that non-retryable status codes return False."""
        non_retryable = [200, 201, 400, 401, 403, 404, 405, 422]
        for code in non_retryable:
            assert is_retryable_status(code) is False, f"{code} should not be retryable"


class TestCalculateDelay:
    """Tests for calculate_delay with exponential backoff."""

    def test_delay_increases_with_attempts(self):
        """Test that delay increases with more attempts."""
        # Get average of multiple samples due to jitter
        samples = 100
        avg_delay_0 = sum(calculate_delay(0) for _ in range(samples)) / samples
        avg_delay_2 = sum(calculate_delay(2) for _ in range(samples)) / samples

        assert avg_delay_2 > avg_delay_0

    def test_delay_capped_at_max(self):
        """Test that delay is capped at max_delay."""
        max_delay = 5.0
        for _ in range(100):
            delay = calculate_delay(10, max_delay=max_delay)
            assert delay <= max_delay

    def test_delay_non_negative(self):
        """Test that delay is always non-negative."""
        for attempt in range(10):
            delay = calculate_delay(attempt)
            assert delay >= 0


class TestRetryRequest:
    """Tests for retry_request."""

    def test_success_on_first_try(self):
        """Test successful request on first attempt."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_func = MagicMock(return_value=mock_response)

        result = retry_request(mock_func, description="test")

        assert result == mock_response
        assert mock_func.call_count == 1

    @patch("paperfind.retry.time.sleep")
    def test_retry_on_connection_error(self, mock_sleep):
        """Test retry on connection error."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_func = MagicMock(
            side_effect=[
                requests.exceptions.ConnectionError("Connection failed"),
                mock_response,
            ]
        )

        result = retry_request(mock_func, max_retries=2, description="test")

        assert result == mock_response
        assert mock_func.call_count == 2
        assert mock_sleep.call_count == 1

    @patch("paperfind.retry.time.sleep")
    def test_retry_on_timeout(self, mock_sleep):
        """Test retry on timeout error."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_func = MagicMock(
            side_effect=[
                requests.exceptions.Timeout("Request timed out"),
                mock_response,
            ]
        )

        result = retry_request(mock_func, max_retries=2, description="test")

        assert result == mock_response
        assert mock_func.call_count == 2

    @patch("paperfind.retry.time.sleep")
    def test_retry_on_retryable_status(self, mock_sleep):
        """Test retry on retryable HTTP status code."""
        mock_response_503 = MagicMock()
        mock_response_503.status_code = 503

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200

        mock_func = MagicMock(side_effect=[mock_response_503, mock_response_200])

        result = retry_request(mock_func, max_retries=2, description="test")

        assert result == mock_response_200
        assert mock_func.call_count == 2

    @patch("paperfind.retry.time.sleep")
    def test_exhausted_retries(self, mock_sleep):
        """Test that exception is raised after exhausting retries."""
        mock_func = MagicMock(
            side_effect=requests.exceptions.ConnectionError("Connection failed")
        )

        with pytest.raises(requests.exceptions.ConnectionError):
            retry_request(mock_func, max_retries=2, description="test")

        assert mock_func.call_count == 3  # Initial + 2 retries

    def test_no_retry_on_400_error(self):
        """Test that 400 errors are not retried."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_func = MagicMock(return_value=mock_response)

        result = retry_request(mock_func, description="test")

        assert result == mock_response
        assert mock_func.call_count == 1

    @patch("paperfind.retry.time.sleep")
    def test_retry_on_rate_limit(self, mock_sleep):
        """Test retry on 429 rate limit."""
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429

        mock_response_200 = MagicMock()
        mock_response_200.status_code = 200

        mock_func = MagicMock(side_effect=[mock_response_429, mock_response_200])

        result = retry_request(mock_func, max_retries=2, description="test")

        assert result == mock_response_200
        assert mock_func.call_count == 2


class TestWithRetryDecorator:
    """Tests for with_retry decorator."""

    def test_success_on_first_try(self):
        """Test decorated function succeeds on first try."""
        @with_retry(max_retries=2, description="test")
        def successful_func():
            return "success"

        assert successful_func() == "success"

    @patch("paperfind.retry.time.sleep")
    def test_retry_on_failure(self, mock_sleep):
        """Test decorator retries on failure."""
        call_count = 0

        @with_retry(max_retries=2, description="test")
        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise requests.exceptions.ConnectionError("Failed")
            return "success"

        result = flaky_func()

        assert result == "success"
        assert call_count == 2

    @patch("paperfind.retry.time.sleep")
    def test_exhausted_retries_raises(self, mock_sleep):
        """Test decorator raises after exhausting retries."""
        @with_retry(max_retries=2, description="test")
        def always_fails():
            raise requests.exceptions.Timeout("Timeout")

        with pytest.raises(requests.exceptions.Timeout):
            always_fails()

    def test_non_retryable_exception_not_retried(self):
        """Test that non-retryable exceptions are not retried."""
        call_count = 0

        @with_retry(max_retries=2, description="test")
        def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            raises_value_error()

        assert call_count == 1  # Should not retry
