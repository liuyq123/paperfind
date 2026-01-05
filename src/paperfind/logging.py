"""Logging configuration for Paperfind."""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

# Default format for log messages
DEFAULT_FORMAT = "%(message)s"
VERBOSE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

# Module-level logger cache
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name.

    Args:
        name: Logger name, typically __name__ from the calling module

    Returns:
        Configured logger instance
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    _loggers[name] = logger
    return logger


def setup_logging(
    level: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). If None, uses
               PAPERFIND_LOG_LEVEL env var or defaults to INFO.
        verbose: If True, use verbose format with timestamps.
    """
    # Determine log level
    if level is None:
        level = os.getenv("PAPERFIND_LOG_LEVEL", "INFO")

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Choose format based on verbosity
    log_format = VERBOSE_FORMAT if verbose else DEFAULT_FORMAT

    # Configure root logger for paperfind
    root_logger = logging.getLogger("paperfind")
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create console handler (stderr for logs)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(numeric_level)
    handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
