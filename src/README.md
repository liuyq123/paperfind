# Internals

Architecture, extension points, and API reference. User-facing docs are in the root [README.md](../README.md).

## Project Structure

```
src/
└── paperfind/            # Main package
    ├── api.py            # FastAPI REST API
    ├── cli.py            # Command-line interface
    ├── config.py         # Configuration and paths
    ├── db.py             # Database abstraction (SQLite/Postgres)
    ├── documents.py      # Document parsing utilities
    ├── embeddings.py     # Embedding provider support
    ├── llm_rerank.py     # LLM-based reranking with user preferences
    ├── logging.py        # Logging configuration
    ├── types.py          # Shared type definitions
    ├── vectorstore.py    # Vector store backends (Chroma/pgvector)
    ├── digest/           # Email digest pipeline
    │   ├── digest.py
    │   ├── email.py
    │   └── template.py
    ├── fetchers/         # Paper fetching modules
    │   ├── db.py
    │   ├── fetch_papers.py
    │   ├── vector.py
    │   ├── sources/
    │   │   ├── arxiv.py
    │   │   ├── biorxiv.py
    │   │   ├── chemrxiv.py
    │   │   └── crossref.py
    │   └── zotero/
    │       ├── api.py
    │       ├── db.py
    │       ├── sync.py
    │       └── vector.py
    └── search/           # Search and recommendation
        ├── formatting.py
        ├── recommend.py
        ├── search.py
        └── utils.py
```

## Architecture Overview

Paperfind:
1. Fetches papers from multiple sources (CrossRef, arXiv, bioRxiv, medRxiv, ChemRxiv)
2. Stores metadata in SQLite or PostgreSQL
3. Builds vector embeddings for semantic search
4. Recommends papers based on similarity to your Zotero library

### Vector Store Abstraction

Backend selection:

```bash
PAPERFIND_VECTOR_STORE=chroma   # default
PAPERFIND_VECTOR_STORE=pgvector
```

Key helpers:

```python
vectordb = get_vector_store(source="daily_papers")
exists = vector_store_exists(source="daily_papers")
embeddings = get_embeddings_from_store(vectordb, ids=["doi1", "doi2"])
results = similarity_search_by_vector(vectordb, embedding, k=10)
```

### Embeddings System

Supported providers: `openai`, `ollama`, `huggingface`. Configure via:

```bash
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

Each provider/model pair gets its own vector store directory/table.

## Database Schemas

### SQLite (default)

**daily_papers.db** (table: `works`)
| Column | Description |
|--------|-------------|
| `doi` | Paper identifier (PRIMARY KEY) |
| `title` | Paper title |
| `authors` | Author list |
| `abstract` | Paper abstract |
| `created_date` | Publication date |
| `type` | Article type (journal-article, preprint, etc.) |
| `source` | Fetch source (crossref, biorxiv, arxiv, etc.) |

**zotero_meta.db** (tables: `libraries`, `collections`, `items`, `item_collections`, `tags`)
| Table | Description |
|-------|-------------|
| `libraries` | Zotero libraries synced (user or group libraries) |
| `collections` | Zotero collections within each library |
| `items` | Papers with metadata (zotero_key, title, authors, abstract, DOI, date, URL) |
| `item_collections` | Many-to-many relationship linking items to collections |
| `tags` | Research tags for organization |

### Postgres (optional)

When `PAPERFIND_DB_URL` is set, data is stored in a single Postgres database with two schemas:

| Schema | Tables | Description |
|--------|--------|-------------|
| `daily` | `works` | Same columns as `daily_papers.db` |
| `zotero` | `libraries`, `collections`, `items`, `item_collections`, `tags` | Same as `zotero_meta.db` |

### ChromaDB (default vector store)

| Directory | Description |
|-----------|-------------|
| `chroma_store_<provider>_<model>/` | Embeddings for fetched papers |
| `zotero_vectors_<provider>_<model>/` | Embeddings for Zotero items |

### pgvector (optional vector store)

| Table | Description |
|-------|-------------|
| `daily.daily_vectors_<provider>_<model>` | Embeddings for fetched papers |
| `zotero.zotero_vectors_<provider>_<model>` | Embeddings for Zotero items |

## API Server

Install and run:

```bash
pip install paperfind[api]
uvicorn paperfind.api:app --reload
```

Interactive docs at `http://localhost:8000/docs`.

Endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/collections` | List Zotero collections |
| GET | `/search?query=...` | Semantic search (supports `scores`, `rag`, `source`) |
| GET | `/papers` | List fetched papers (paginated) |
| GET | `/recommend` | Get paper recommendations |
| POST | `/sync` | Trigger Zotero library sync (background job) |
| POST | `/embed` | Embed Zotero items (optional: `collection`) |
| POST | `/fetch` | Trigger paper fetching (background job) |
| GET | `/jobs/{job_id}` | Check background job status |

## Development

### Adding a New Paper Source

1. Implement a fetcher in `src/paperfind/fetchers/sources/` that returns a list of `PaperDict`.
2. Add the source to `fetch_all()` in `src/paperfind/fetchers/fetch_papers.py`.
3. Add the source name to CLI `--source` choices in `src/paperfind/cli.py`.
4. Update README docs and add tests.

Required `PaperDict` fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `doi` | str | Yes | Unique identifier (DOI or similar) |
| `title` | str | Yes | Paper title |
| `authors` | str | No | Comma-separated author names |
| `abstract` | str | Yes | Paper abstract (used for embeddings) |
| `created_date` | str | No | Publication date (YYYY-MM-DD) |
| `type` | str | No | Paper type (preprint, journal-article) |
| `source` | str | Yes | Source identifier for filtering |
