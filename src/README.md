# Project Structure

```
src/
└── paperfind/            # Main package
    ├── __init__.py
    ├── cli.py            # Command-line interface
    ├── api.py            # FastAPI REST API
    ├── config.py         # Configuration and paths
    ├── db.py             # Database abstraction (SQLite/Postgres)
    ├── documents.py      # Document parsing utilities
    ├── embeddings.py     # Embedding provider support
    ├── logging.py        # Logging configuration
    ├── rerank.py         # Cross-encoder reranking
    ├── types.py          # Shared type definitions
    ├── vectorstore.py    # Vector store backends (Chroma/pgvector)
    ├── fetchers/         # Paper fetching modules
    │   ├── db.py
    │   ├── fetch_papers.py
    │   ├── vector.py
    │   ├── sources/
    │   │   ├── arxiv.py
    │   │   ├── biorxiv.py
    │   │   └── crossref.py
    │   └── zotero/
    │       ├── api.py
    │       ├── db.py
    │       ├── sync.py
    │       └── vector.py
    ├── digest/           # Email digest pipeline
    │   ├── digest.py
    │   ├── email.py
    │   └── template.py
    └── search/           # Search and recommendation
        ├── recommend.py
        ├── search.py
        ├── formatting.py
        └── utils.py
```

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
| `items` | Papers with metadata (zotero_key, title, authors, abstract, DOI, date, URL) - unique per library |
| `item_collections` | Many-to-many relationship linking items to collections |
| `tags` | Research tags for organization |

### Postgres (optional)

When `PAPERFIND_DB_URL` is set, data is stored in a single Postgres database with two schemas:

| Schema | Tables | Description |
|--------|--------|-------------|
| `daily` | `works` | Same columns as `daily_papers.db` |
| `zotero` | `libraries`, `collections`, `items`, `item_collections`, `tags` | Same as `zotero_meta.db` |

### ChromaDB (default vector store)

When using the default `PAPERFIND_VECTOR_STORE=chroma`, embeddings are stored in local directories:

| Directory | Description |
|-----------|-------------|
| `chroma_store_<provider>_<model>/` | Embeddings for fetched papers |
| `zotero_vectors_<provider>_<model>/` | Embeddings for Zotero items |

ChromaDB stores data in SQLite files within these directories (`chroma.sqlite3`).

### pgvector (optional vector store)

When `PAPERFIND_VECTOR_STORE=pgvector`, embeddings are stored in Postgres tables:

| Table | Description |
|-------|-------------|
| `daily.daily_vectors_<provider>_<model>` | Embeddings for fetched papers |
| `zotero.zotero_vectors_<provider>_<model>` | Embeddings for Zotero items |

Each vector table has columns: `id`, `embedding`, `document`, `metadata`

## API Server

Paperfind includes a REST API for programmatic access.

### Setup

```bash
pip install paperfind[api]
uvicorn paperfind.api:app --reload
```

Interactive API docs available at `http://localhost:8000/docs`.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/collections` | List Zotero collections |
| GET | `/search?query=...` | Semantic search (supports `scores`, `rag`, `source` params) |
| GET | `/papers` | List fetched papers (paginated) |
| GET | `/recommend` | Get paper recommendations |
| POST | `/sync` | Trigger Zotero library sync (background job) |
| POST | `/embed` | Embed Zotero items for semantic search (optional: `collection`) |
| POST | `/fetch` | Trigger paper fetching (background job) |
| GET | `/jobs/{job_id}` | Check background job status |

### Examples

```bash
# Search papers
curl "http://localhost:8000/search?query=machine+learning&k=5"

# Get recommendations
curl "http://localhost:8000/recommend?k=10"

# Sync your Zotero library
curl -X POST "http://localhost:8000/sync"

# Embed all items
curl -X POST "http://localhost:8000/embed"

# Embed a specific collection
curl -X POST "http://localhost:8000/embed?collection=my+research"

# Start a fetch job
curl -X POST "http://localhost:8000/fetch?days=3"
# Returns: {"job_id": "abc-123", "status": "pending", "job_type": "fetch"}

# Check job status
curl "http://localhost:8000/jobs/abc-123"
```
