"""
Embedding provider abstraction layer.

Supports: OpenAI, Ollama, HuggingFace (sentence-transformers)
"""

import os
import re
from typing import Optional

from langchain_core.embeddings import Embeddings


# Default models per provider
DEFAULT_MODELS = {
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text",
    "huggingface": "all-MiniLM-L6-v2",
}


def get_embedding_provider() -> str:
    """Get the configured embedding provider."""
    return os.getenv("EMBEDDING_PROVIDER", "openai").lower()


def get_embedding_model() -> str:
    """Get the configured embedding model, with provider-specific defaults."""
    provider = get_embedding_provider()
    default_model = DEFAULT_MODELS.get(provider, DEFAULT_MODELS["openai"])
    return os.getenv("EMBEDDING_MODEL", default_model)


def sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in directory paths."""
    return re.sub(r'[/\\:*?"<>|]', "_", model)


def get_embeddings(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs,
) -> Embeddings:
    """
    Factory function to get the appropriate embedding provider.

    Args:
        model: Model name (optional, uses EMBEDDING_MODEL env var if not provided)
        provider: Provider name (optional, uses EMBEDDING_PROVIDER env var if not provided)
        **kwargs: Additional provider-specific arguments

    Returns:
        LangChain Embeddings instance

    Raises:
        ValueError: If provider is not supported
        ImportError: If required provider package is not installed
    """
    provider = provider or get_embedding_provider()
    model = model or get_embedding_model()

    if provider == "openai":
        return _get_openai_embeddings(model, **kwargs)
    elif provider == "ollama":
        return _get_ollama_embeddings(model, **kwargs)
    elif provider == "huggingface":
        return _get_huggingface_embeddings(model, **kwargs)
    else:
        raise ValueError(
            f"Unsupported embedding provider: {provider}. "
            f"Supported: openai, ollama, huggingface"
        )


def _get_openai_embeddings(model: str, **kwargs) -> Embeddings:
    """Get OpenAI embeddings."""
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        raise ImportError(
            "OpenAI embeddings require langchain-openai. "
            "Install with: pip install langchain-openai"
        )

    defaults = {"chunk_size": 50}
    defaults.update(kwargs)
    return OpenAIEmbeddings(model=model, **defaults)


def _get_ollama_embeddings(model: str, **kwargs) -> Embeddings:
    """Get Ollama embeddings."""
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        raise ImportError(
            "Ollama embeddings require langchain-ollama. "
            "Install with: pip install langchain-ollama"
        )

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return OllamaEmbeddings(model=model, base_url=base_url, **kwargs)


def _get_huggingface_embeddings(model: str, **kwargs) -> Embeddings:
    """Get HuggingFace sentence-transformers embeddings."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        raise ImportError(
            "HuggingFace embeddings require langchain-huggingface. "
            "Install with: pip install langchain-huggingface sentence-transformers"
        )

    return HuggingFaceEmbeddings(model_name=model, **kwargs)
