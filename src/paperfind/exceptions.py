"""Custom exception classes for Paperfind."""


class PaperfindError(Exception):
    """Base exception for all Paperfind errors."""

    pass


class ConfigError(PaperfindError):
    """Raised when there's a configuration error.

    Examples:
        - Missing required environment variable
        - Invalid configuration value
        - Missing .env file
    """

    pass


class FetcherError(PaperfindError):
    """Raised when a paper fetcher encounters an error.

    Examples:
        - Network error during API call
        - Invalid response from API
        - Rate limiting
    """

    def __init__(self, source: str, message: str):
        self.source = source
        self.message = message
        super().__init__(f"[{source}] {message}")


class VectorStoreError(PaperfindError):
    """Raised when there's an error with vector store operations.

    Examples:
        - Failed to initialize vector store
        - Failed to add/query embeddings
        - Backend connection error
    """

    pass


class EmbeddingError(PaperfindError):
    """Raised when there's an error with embedding operations.

    Examples:
        - Failed to generate embeddings
        - Invalid embedding provider
        - API error from embedding service
    """

    pass


class ZoteroError(PaperfindError):
    """Raised when there's an error with Zotero operations.

    Examples:
        - Failed to sync library
        - Invalid Zotero credentials
        - API rate limiting
    """

    pass
