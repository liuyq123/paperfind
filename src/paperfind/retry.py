"""Retry utilities for handling transient failures."""

import random
import time
from functools import wraps
from typing import Callable, Tuple, Type, TypeVar

import requests

from paperfind.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 30.0  # seconds
DEFAULT_EXPONENTIAL_BASE = 2

# Exceptions that should trigger a retry
RETRYABLE_EXCEPTIONS: Tuple[Type[Exception], ...] = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def is_retryable_status(status_code: int) -> bool:
    """Check if an HTTP status code should trigger a retry.

    Retryable status codes:
    - 429: Too Many Requests (rate limited)
    - 500: Internal Server Error
    - 502: Bad Gateway
    - 503: Service Unavailable
    - 504: Gateway Timeout
    """
    return status_code in {429, 500, 502, 503, 504}


def calculate_delay(
    attempt: int,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    exponential_base: int = DEFAULT_EXPONENTIAL_BASE,
) -> float:
    """Calculate delay with exponential backoff and jitter.

    Uses exponential backoff with full jitter to prevent thundering herd.
    """
    exponential_delay = base_delay * (exponential_base ** attempt)
    capped_delay = min(exponential_delay, max_delay)
    # Full jitter: random value between 0 and capped_delay
    return random.uniform(0, capped_delay)


def retry_request(
    func: Callable[[], requests.Response],
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    description: str = "request",
) -> requests.Response:
    """Execute a request function with retry logic.

    Args:
        func: A callable that returns a requests.Response
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        max_delay: Maximum delay in seconds
        description: Description for logging (e.g., "arXiv API")

    Returns:
        The successful response

    Raises:
        requests.RequestException: If all retries fail
    """
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            response = func()

            # Check for retryable HTTP status codes
            if is_retryable_status(response.status_code):
                if attempt < max_retries:
                    delay = calculate_delay(attempt, base_delay, max_delay)
                    logger.warning(
                        f"{description} returned {response.status_code}, "
                        f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})"
                    )
                    time.sleep(delay)
                    continue
                else:
                    # Final attempt failed, raise the status
                    response.raise_for_status()

            return response

        except RETRYABLE_EXCEPTIONS as exc:
            last_exception = exc
            if attempt < max_retries:
                delay = calculate_delay(attempt, base_delay, max_delay)
                logger.warning(
                    f"{description} failed: {exc}, "
                    f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})"
                )
                time.sleep(delay)
            else:
                logger.error(f"{description} failed after {max_retries + 1} attempts: {exc}")
                raise

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception
    raise requests.RequestException(f"{description} failed unexpectedly")


def with_retry(
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    description: str = "operation",
):
    """Decorator to add retry logic to a function.

    The decorated function should raise an exception on failure.
    Only RETRYABLE_EXCEPTIONS will trigger a retry.

    Example:
        @with_retry(max_retries=3, description="fetch data")
        def fetch_data():
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return response.json()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as exc:
                    last_exception = exc
                    if attempt < max_retries:
                        delay = calculate_delay(attempt, base_delay, max_delay)
                        logger.warning(
                            f"{description} failed: {exc}, "
                            f"retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries + 1})"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"{description} failed after {max_retries + 1} attempts: {exc}")
                        raise

            # Should not reach here
            if last_exception:
                raise last_exception
            raise RuntimeError(f"{description} failed unexpectedly")

        return wrapper

    return decorator
