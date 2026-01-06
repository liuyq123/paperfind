"""
Cross-encoder reranking utilities.

Default model: mixedbread-ai/mxbai-rerank-base-v1
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, List, Optional, Sequence, Tuple

DEFAULT_RERANK_MODEL = "mixedbread-ai/mxbai-rerank-base-v1"


def get_rerank_model() -> str:
    """Get the configured rerank model."""
    return os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL)


@lru_cache(maxsize=None)
def _get_cross_encoder(model: str) -> Any:
    """Get or create a cached CrossEncoder instance."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "Reranking requires sentence-transformers. "
            "Install with: pip install paperfind[huggingface]"
        ) from exc

    return CrossEncoder(model)


def rerank_pairs(
    pairs: Sequence[Tuple[str, str]],
    model: Optional[str] = None,
) -> List[float]:
    """
    Rerank (query, document) pairs using a cross-encoder.

    Returns a list of scores aligned to the input order.
    """
    if not pairs:
        return []

    rerank_model = model or get_rerank_model()
    encoder = _get_cross_encoder(rerank_model)
    scores = encoder.predict(list(pairs))

    if hasattr(scores, "tolist"):
        return scores.tolist()
    return list(scores)
