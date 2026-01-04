"""Vector store abstraction for Chroma (default) and optional pgvector backend."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.base import VectorStore

from paperfind.config import get_chroma_store_dir, get_zotero_vectors_dir
from paperfind.db import DAILY_SCHEMA, ZOTERO_SCHEMA, db_url, is_postgres, table_exists
from paperfind.embeddings import (
    get_embedding_model,
    get_embedding_provider,
    get_embeddings,
    sanitize_model_name,
)
from paperfind.logging import get_logger

logger = get_logger(__name__)

VECTOR_STORE_ENV = "PAPERFIND_VECTOR_STORE"
DEFAULT_VECTOR_STORE = "chroma"

ZOTERO_COLLECTION = "zotero_all"


def get_vector_store_backend() -> str:
    """Return the configured vector store backend."""
    value = os.getenv(VECTOR_STORE_ENV) or os.getenv("VECTOR_STORE") or DEFAULT_VECTOR_STORE
    backend = value.lower()
    if backend not in {"chroma", "pgvector"}:
        logger.warning(f"Unknown vector store '{value}', defaulting to '{DEFAULT_VECTOR_STORE}'.")
        return DEFAULT_VECTOR_STORE
    return backend


def using_pgvector() -> bool:
    """Return True if pgvector backend is enabled."""
    return get_vector_store_backend() == "pgvector"


def _sanitize_identifier(value: str) -> str:
    sanitized = sanitize_model_name(value)
    sanitized = re.sub(r"[^A-Za-z0-9_]+", "_", sanitized).strip("_")
    return sanitized.lower() or "default"


def _truncate_identifier(value: str, max_len: int = 63) -> str:
    if len(value) <= max_len:
        return value
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]
    return f"{value[: max_len - 9]}_{digest}"


def _vector_suffix() -> str:
    provider = get_embedding_provider()
    model = get_embedding_model()
    return _sanitize_identifier(f"{provider}_{model}")


def _vector_schema(source: str) -> str:
    return ZOTERO_SCHEMA if source == "zotero" else DAILY_SCHEMA


def _vector_table_name(source: str) -> str:
    prefix = "zotero_vectors" if source == "zotero" else "daily_vectors"
    name = f"{prefix}_{_vector_suffix()}"
    return _truncate_identifier(name)


def _pgvector_connect():
    if not is_postgres():
        raise ValueError("PAPERFIND_DB_URL is required for pgvector backend.")
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise ImportError(
            "Postgres support requires psycopg. Install with: pip install paperfind[postgres]"
        ) from exc
    try:
        from pgvector.psycopg import register_vector
    except ImportError as exc:
        raise ImportError(
            "pgvector support requires the pgvector package. "
            "Install with: pip install paperfind[postgres]"
        ) from exc

    conn = psycopg.connect(db_url(), row_factory=dict_row)
    register_vector(conn)
    return conn


def _pgvector_table_exists(source: str) -> bool:
    if not is_postgres():
        return False
    try:
        conn = _pgvector_connect()
    except ImportError as exc:
        logger.error(str(exc))
        return False
    except Exception as exc:
        logger.error(f"Failed to connect to Postgres: {exc}")
        return False
    schema = _vector_schema(source)
    table = _vector_table_name(source)
    exists = table_exists(conn, schema, table)
    conn.close()
    return exists


def vector_store_exists(source: str = "daily_papers") -> bool:
    """Return True if the configured vector store exists for the source."""
    backend = get_vector_store_backend()
    if backend == "chroma":
        if source == "zotero":
            return Path(get_zotero_vectors_dir()).exists()
        return Path(get_chroma_store_dir()).exists()
    if backend == "pgvector":
        return _pgvector_table_exists(source)
    return False


def get_vector_store(source: str = "daily_papers") -> VectorStore:
    """Return a vector store instance for the requested source."""
    backend = get_vector_store_backend()
    embeddings = get_embeddings()

    if backend == "chroma":
        try:
            from langchain_chroma import Chroma
        except ImportError as exc:
            raise ImportError(
                "Chroma support requires langchain-chroma. Install with: pip install langchain-chroma"
            ) from exc

        if source == "zotero":
            vectors_dir = get_zotero_vectors_dir()
            Path(vectors_dir).mkdir(parents=True, exist_ok=True)
            return Chroma(
                embedding_function=embeddings,
                persist_directory=vectors_dir,
                collection_name=ZOTERO_COLLECTION,
            )
        vectors_dir = get_chroma_store_dir()
        Path(vectors_dir).mkdir(parents=True, exist_ok=True)
        return Chroma(
            embedding_function=embeddings,
            persist_directory=vectors_dir,
        )

    if backend == "pgvector":
        return PGVectorStore(embedding_function=embeddings, source=source)

    raise ValueError(f"Unsupported vector store backend: {backend}")


class PGVectorStore(VectorStore):
    """Minimal pgvector-backed VectorStore implementation."""

    def __init__(self, embedding_function: Embeddings, source: str) -> None:
        self.embedding_function = embedding_function
        self.source = source
        self.schema = _vector_schema(source)
        self.table = _vector_table_name(source)
        self._index_created = False
        self._ensure_schema_and_table()

    @property
    def embeddings(self) -> Embeddings:
        """Return embedding function (required by as_retriever)."""
        return self.embedding_function

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "PGVectorStore":
        source = kwargs.get("source", "daily_papers")
        store = cls(embedding_function=embedding, source=source)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        text_list = list(texts)
        if not text_list:
            return []

        metadata_list = metadatas if metadatas is not None else [{} for _ in text_list]
        if len(metadata_list) != len(text_list):
            raise ValueError("metadatas length must match texts length")

        if ids is not None and len(ids) != len(text_list):
            raise ValueError("ids length must match texts length")
        id_list = ids or [str(uuid.uuid4()) for _ in text_list]
        embeddings = self.embedding_function.embed_documents(text_list)

        records = [
            (id_value, embedding, text, json.dumps(metadata or {}))
            for id_value, embedding, text, metadata in zip(id_list, embeddings, text_list, metadata_list)
        ]

        conn = self._connect()
        cur = conn.cursor()
        qualified = self._qualified_table()
        cur.executemany(
            f"""
            INSERT INTO {qualified} (id, embedding, document, metadata)
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                document = EXCLUDED.document,
                metadata = EXCLUDED.metadata;
            """,
            records,
        )
        conn.commit()
        conn.close()
        return id_list

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        results = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in results]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        self._ensure_index()
        embedding = self.embedding_function.embed_query(query)
        return self.similarity_search_by_vector_with_score(embedding, k=k, **kwargs)

    def similarity_search_by_vector_with_score(
        self,
        embedding: List[float],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search using a raw embedding vector."""
        self._ensure_index()
        filter_clause, filter_params = _build_filter_clause(kwargs.get("filter"))

        conn = self._connect()
        cur = conn.cursor()
        qualified = self._qualified_table()
        cur.execute(
            f"""
            SELECT document, metadata, (embedding <=> %s) AS distance
            FROM {qualified}
            {filter_clause}
            ORDER BY distance ASC
            LIMIT %s
            """,
            [embedding, *filter_params, k],
        )
        rows = cur.fetchall()
        conn.close()

        results: List[Tuple[Document, float]] = []
        for row in rows:
            document = row["document"]
            metadata = _normalize_metadata(row["metadata"])
            distance = float(row["distance"])
            results.append((Document(page_content=document, metadata=metadata), distance))
        return results

    def get_embeddings_by_ids(self, ids: List[str]) -> Dict[str, List[float]]:
        """Return {id: embedding} for requested IDs."""
        if not ids:
            return {}

        conn = self._connect()
        cur = conn.cursor()
        qualified = self._qualified_table()
        placeholders = ", ".join(["%s"] * len(ids))
        cur.execute(
            f"SELECT id, embedding FROM {qualified} WHERE id IN ({placeholders})",
            ids,
        )
        rows = cur.fetchall()
        conn.close()

        return {row["id"]: list(row["embedding"]) for row in rows}

    def delete(self, ids: Optional[List[str]] = None, where: Optional[dict] = None) -> None:
        conn = self._connect()
        cur = conn.cursor()
        qualified = self._qualified_table()

        if ids:
            placeholders = ", ".join(["%s"] * len(ids))
            cur.execute(f"DELETE FROM {qualified} WHERE id IN ({placeholders})", ids)
        elif where:
            clause, params = _build_filter_clause(where)
            cur.execute(f"DELETE FROM {qualified} {clause}", params)
        else:
            cur.execute(f"DELETE FROM {qualified}")

        conn.commit()
        conn.close()

    def count(self, where: Optional[dict] = None) -> int:
        conn = self._connect()
        cur = conn.cursor()
        qualified = self._qualified_table()
        clause, params = _build_filter_clause(where)
        cur.execute(f"SELECT COUNT(*) AS cnt FROM {qualified} {clause}", params)
        result = cur.fetchone()["cnt"]
        conn.close()
        return int(result)

    def list_ids(self) -> List[str]:
        """Return all document ids in the vector store."""
        conn = self._connect()
        cur = conn.cursor()
        qualified = self._qualified_table()
        cur.execute(f"SELECT id FROM {qualified}")
        rows = cur.fetchall()
        conn.close()
        return [row["id"] for row in rows]

    def has_id(self, item_id: str) -> bool:
        """Return True if a document id exists in the vector store."""
        conn = self._connect()
        cur = conn.cursor()
        qualified = self._qualified_table()
        cur.execute(f"SELECT 1 FROM {qualified} WHERE id = %s LIMIT 1", (item_id,))
        exists = cur.fetchone() is not None
        conn.close()
        return exists

    def _connect(self):
        return _pgvector_connect()

    def _qualified_table(self) -> str:
        return f"{self.schema}.{self.table}"

    def _ensure_schema_and_table(self) -> None:
        conn = self._connect()
        cur = conn.cursor()
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")

        try:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except Exception as exc:
            conn.close()
            raise RuntimeError(
                "Failed to enable pgvector extension. "
                "Make sure `CREATE EXTENSION vector;` succeeds."
            ) from exc

        if not table_exists(conn, self.schema, self.table):
            dim = self._embedding_dim()
            qualified = self._qualified_table()
            cur.execute(
                f"""
                CREATE TABLE {qualified} (
                    id TEXT PRIMARY KEY,
                    embedding VECTOR({dim}),
                    document TEXT NOT NULL,
                    metadata JSONB
                );
                """
            )

        conn.commit()
        conn.close()

    def _embedding_dim(self) -> int:
        sample = self.embedding_function.embed_query("dimension")
        return len(sample)

    def _index_name(self) -> str:
        return _truncate_identifier(f"{self.table}_embedding_idx")

    def _index_exists(self, conn) -> bool:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM pg_indexes WHERE schemaname = %s AND indexname = %s",
            (self.schema, self._index_name()),
        )
        return cur.fetchone() is not None

    def _ensure_index(self) -> None:
        """Create IVFFlat index if data exists and index hasn't been created."""
        if self._index_created:
            return

        conn = self._connect()
        if self._index_exists(conn):
            self._index_created = True
            conn.close()
            return

        # Check if there's data to index
        cur = conn.cursor()
        qualified = self._qualified_table()
        cur.execute(f"SELECT COUNT(*) AS cnt FROM {qualified}")
        count = cur.fetchone()["cnt"]

        if count > 0:
            index_name = self._index_name()
            logger.debug(f"Creating IVFFlat index on {qualified} ({count} rows)...")
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {qualified}
                USING ivfflat (embedding vector_cosine_ops);
                """
            )
            conn.commit()
            self._index_created = True

        conn.close()


def _build_filter_clause(metadata_filter: Optional[Dict[str, Any]]) -> Tuple[str, List[Any]]:
    if not metadata_filter:
        return "", []
    return "WHERE metadata @> %s::jsonb", [json.dumps(metadata_filter)]


def get_embeddings_from_store(
    vectordb: VectorStore,
    ids: List[str],
) -> Dict[str, List[float]]:
    """
    Get embeddings by ID from any vector store backend.

    Returns {id: embedding} for found IDs.
    """
    if not ids:
        return {}

    # PGVectorStore has native method
    if isinstance(vectordb, PGVectorStore):
        return vectordb.get_embeddings_by_ids(ids)

    # Chroma: use _collection.get()
    if hasattr(vectordb, "_collection"):
        try:
            result = vectordb._collection.get(ids=ids, include=["embeddings"])
            if result and result.get("ids") and result.get("embeddings"):
                return dict(zip(result["ids"], result["embeddings"]))
        except Exception as exc:
            logger.warning(f"Failed to get embeddings from Chroma: {exc}")

    return {}


def similarity_search_by_vector(
    vectordb: VectorStore,
    embedding: List[float],
    k: int = 4,
    **kwargs: Any,
) -> List[Tuple[Document, float]]:
    """
    Search by raw embedding vector in any vector store backend.

    Returns list of (Document, score) tuples.
    """
    # PGVectorStore has native method
    if isinstance(vectordb, PGVectorStore):
        return vectordb.similarity_search_by_vector_with_score(embedding, k=k, **kwargs)

    # Chroma/LangChain stores have similarity_search_by_vector
    if hasattr(vectordb, "similarity_search_by_vector_with_relevance_scores"):
        return vectordb.similarity_search_by_vector_with_relevance_scores(embedding, k=k, **kwargs)

    # Fallback: some stores return without scores
    if hasattr(vectordb, "similarity_search_by_vector"):
        docs = vectordb.similarity_search_by_vector(embedding, k=k, **kwargs)
        return [(doc, 0.0) for doc in docs]

    raise NotImplementedError(f"Vector store {type(vectordb)} does not support similarity_search_by_vector")


def _normalize_metadata(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return dict(value)
