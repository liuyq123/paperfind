# Project Structure

```
src/
└── paperfind/            # Main package
    ├── __init__.py
    ├── cli.py            # Command-line interface
    ├── config.py         # Configuration and paths
    ├── db.py             # Database abstraction (SQLite/Postgres)
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

**zotero_meta.db** (tables: `projects`, `items`, `tags`)
| Table | Description |
|-------|-------------|
| `projects` | Zotero collections/libraries synced |
| `items` | Papers with metadata (zotero_key, title, authors, abstract, DOI, date, URL) |
| `tags` | Research tags for organization |

### Postgres (optional)

When `PAPERFIND_DB_URL` is set, data is stored in a single Postgres database with two schemas:

| Schema | Tables | Description |
|--------|--------|-------------|
| `daily` | `works` | Same columns as `daily_papers.db` |
| `zotero` | `projects`, `items`, `tags` | Same as `zotero_meta.db` |

### pgvector Embeddings (optional)

When `PAPERFIND_VECTOR_STORE=pgvector`, embeddings are stored in Postgres tables:

| Table | Description |
|-------|-------------|
| `daily.daily_vectors_<provider>_<model>` | Embeddings for fetched papers |
| `zotero.zotero_vectors_<provider>_<model>` | Embeddings for Zotero items |

Each vector table has columns: `id`, `embedding`, `document`, `metadata`
