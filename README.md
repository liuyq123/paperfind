# Paperfind

A paper recommendation system that discovers relevant papers and preprints based on your Zotero library. It fetches metadata from CrossRef, bioRxiv, medRxiv, and arXiv, then uses semantic search and cross-encoder reranking to recommend papers similar to your existing research interests.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [From Source (Recommended)](#from-source-recommended)
  - [Using pip](#using-pip)
- [Configuration](#configuration)
  - [Step 1: Create your `.env` file](#step-1-create-your-env-file)
  - [Step 2: Verify configuration](#step-2-verify-configuration)
- [Usage](#usage)
  - [Quick Start: Get Today's Recommendations](#quick-start-get-todays-recommendations)
  - [Sync Zotero Library](#sync-zotero-library)
  - [Embed Collections](#embed-collections)
  - [Get Recommendations](#get-recommendations)
  - [Fetch Papers](#fetch-papers)
  - [Email Digest](#email-digest)
  - [Semantic Search](#semantic-search)
  - [API Server (optional)](#api-server-optional)
  - [Embedding Providers](#embedding-providers)
- [Data Storage](#data-storage)
  - [What Happens on Repeated Runs](#what-happens-on-repeated-runs)

## Features

- **Paper Recommendations**: Get daily paper recommendations based on your Zotero library
- **Paper Harvesting**: Fetches paper metadata (title, authors, abstract) from CrossRef, bioRxiv, medRxiv, and arXiv
- **Zotero Integration**: Syncs with your personal Zotero library to understand your research interests
- **Semantic Search**: Search across papers using OpenAI embeddings and ChromaDB
- **RAG Pipeline**: Ask questions about your paper collection using GPT-4

## Installation

### From Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/liuyq123/paperfind.git
cd paperfind

# Install in editable mode
pip install -e .
```

After installation, the `paperfind` command will be available:

```bash
paperfind --help
```

### Using pip

```bash
pip install paperfind
```

#### Optional Dependencies

| Extra | Install Command | Description |
|-------|-----------------|-------------|
| `ollama` | `pip install paperfind[ollama]` | Local embeddings via [Ollama](https://ollama.ai) |
| `huggingface` | `pip install paperfind[huggingface]` | Local embeddings via sentence-transformers |
| `all-embeddings` | `pip install paperfind[all-embeddings]` | Both Ollama and HuggingFace support |
| `postgres` | `pip install paperfind[postgres]` | PostgreSQL + pgvector backend |
| `api` | `pip install paperfind[api]` | FastAPI REST server |
| `dev` | `pip install paperfind[dev]` | Development tools (pytest, black, ruff) |

You can combine multiple extras:

```bash
pip install paperfind[postgres,api,ollama]
```

## Configuration

### Step 1: Create your `.env` file

Create a `.env` file in `~/.paperfind/` (recommended) or your current working directory:

```bash
# Create the data directory
mkdir -p ~/.paperfind

# Copy the example .env file and edit it
cp .env.example ~/.paperfind/.env
```

Edit `~/.paperfind/.env` and fill in your keys and settings. See `.env.example` for a complete list with inline comments describing each variable.

To use pgvector, set `PAPERFIND_VECTOR_STORE=pgvector`, ensure `PAPERFIND_DB_URL` is set, and enable the extension in your database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Step 2: Verify configuration

```bash
paperfind config
```

This shows your data directory path. Your `.env` file should be in that directory or your current working directory.

## Usage

### Quick Start: Get Today's Recommendations

```bash
# 1. Sync your Zotero library (syncs entire library)
paperfind sync

# 2. Embed a specific collection for semantic search
paperfind embed "my research collection"

# 3. Fetch today's papers and build embeddings
paperfind fetch --rebuild-vectors

# 4. Get personalized recommendations based on your Zotero library
paperfind recommend
```

### Sync Zotero Library

Sync your Zotero library to get personalized recommendations. The sync command always syncs your **entire library**, storing each item once and tracking collection memberships via a many-to-many relationship.

```bash
# List available collections in your Zotero library
paperfind sync --list-collections

# Sync your entire library (items, collections, and memberships)
paperfind sync
```

### Embed Collections

After syncing, embed specific collections for semantic search. Embeddings are keyed by Zotero item key, so each paper is only embedded once even if it appears in multiple collections.

```bash
# Embed items in a specific collection
paperfind embed "active learning"

# Re-embed all items (ignore existing embeddings)
paperfind embed "active learning" --force
```

### Get Recommendations

Find papers similar to your Zotero library:

```bash
# Get top 10 recommendations (default)
paperfind recommend

# Get more recommendations
paperfind recommend -k 20

# Recommendations based on a specific collection
paperfind recommend --collection "active learning"

# Save recommendations to markdown file
paperfind recommend -o recommendations.md

# Reranking is enabled by default; tune candidate pool or disable if needed
paperfind recommend --rerank-candidates 50
paperfind recommend --no-rerank
```

The markdown file includes title, authors, abstract, date, source, and DOI links for each paper.
Reranking uses the cross-encoder model in `RERANK_MODEL` (default: `mixedbread-ai/mxbai-rerank-base-v1`).
Rerank scores are raw cross-encoder scores where higher is better.

### Fetch Papers

Fetch papers from all sources with a single command:

```bash
# Fetch today's papers from all sources (CrossRef, bioRxiv, medRxiv, arXiv)
paperfind fetch

# Fetch last 7 days (including today) and rebuild vector embeddings
paperfind fetch --days 7 --rebuild-vectors

# Fetch from specific sources only
paperfind fetch --source arxiv --source biorxiv

# Fetch bioRxiv/medRxiv with category filters
paperfind fetch --biorxiv-category bioinformatics --medrxiv-category genomics

# Only rebuild vectors (no fetching)
paperfind fetch --vectors-only
```

Example (category-filtered fetch):

```bash
paperfind fetch --days 3 --biorxiv-category synthetic-biology
```

**Options:**

| Flag | Description |
|------|-------------|
| `--days N` | Number of days to fetch (default: 1) |
| `--source` | Specific source(s): `crossref`, `biorxiv`, `medrxiv`, `arxiv` |
| `--biorxiv-category` | Filter bioRxiv results by category (default: all) |
| `--medrxiv-category` | Filter medRxiv results by category (default: all) |
| `--rebuild-vectors` | Rebuild vector embeddings after fetching |
| `--vectors-only` | Only rebuild vectors, skip fetching |

**Sources and categories:**
- **CrossRef**: Journal articles and preprints with DOIs
- **bioRxiv/medRxiv**: Life science preprints (categories: `bioinformatics`, `biochemistry`, `pharmacology-and-toxicology`, `systems-biology`, `synthetic-biology`, `molecular-biology`, `cell-biology`, `genomics`, `biophysics`). The authoritative list lives in `BIORXIV_CATEGORIES` in `src/paperfind/fetchers/sources/biorxiv.py`.
- **arXiv**: Categories `q-bio.BM`, `q-bio.QM`, `cs.LG`, `cs.AI`, `stat.ML`

### Email Digest

Send a scheduled email with the latest recommendations:

```bash
# Send today's digest email
paperfind digest

# Preview the email without sending
paperfind digest --dry-run

# Include the last 7 days of papers in the digest
paperfind digest --days 7

# Include more recommendations in the email
paperfind digest -k 20

# Skip fetching new papers before generating the digest
paperfind digest --skip-fetch
```

**Required SMTP settings**

Email delivery requires these `.env` entries:

```
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=your_smtp_user
SMTP_PASSWORD=your_smtp_password
EMAIL_FROM=paperfind@example.com
EMAIL_TO=you@example.com,teammate@example.com
```

`EMAIL_TO` accepts a comma-separated list of recipient addresses (no spaces required).

**Note for Gmail users:** If you have 2-factor authentication enabled, use an [App Password](https://myaccount.google.com/apppasswords) instead of your regular password.

**Scheduled runs with GitHub Actions**

If you want the digest to run on a schedule, add a GitHub Actions workflow that runs
`paperfind digest` on a cron. Store the same SMTP and API credentials as repository or
organization secrets, then load them as environment variables in the workflow.

### Semantic Search

Search papers or ask questions using RAG:

```bash
# Basic semantic search
paperfind search "deep learning ligand discovery"

# Search with more results
paperfind search "molecular docking" -k 10

# Show similarity scores
paperfind search "virtual screening" --scores

# Search Zotero library instead of daily papers
paperfind search "active learning" -s zotero

# Ask a question using RAG (Retrieval-Augmented Generation)
paperfind search "What methods are used for ultra-large library screening?" --rag
```

### API Server (optional)

Run the REST API server:

```bash
pip install paperfind[api]
uvicorn paperfind.api:app --reload
```

**Available Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/collections` | List Zotero collections |
| GET | `/search?query=...` | Semantic search (supports `scores`, `rag`, `source` params) |
| GET | `/papers` | List fetched papers (paginated) |
| GET | `/recommend` | Get paper recommendations |
| POST | `/sync` | Trigger Zotero library sync (background job) |
| POST | `/embed?collection=...` | Embed a collection for semantic search (background job) |
| POST | `/fetch` | Trigger paper fetching (background job) |
| GET | `/jobs/{job_id}` | Check background job status |

**Example usage:**

```bash
# Search papers
curl "http://localhost:8000/search?query=machine+learning&k=5"

# Get recommendations
curl "http://localhost:8000/recommend?k=10"

# Sync your Zotero library
curl -X POST "http://localhost:8000/sync"

# Embed a collection
curl -X POST "http://localhost:8000/embed?collection=my+research"

# Start a fetch job
curl -X POST "http://localhost:8000/fetch?days=3"
# Returns: {"job_id": "abc-123", "status": "pending", "job_type": "fetch"}

# Check job status
curl "http://localhost:8000/jobs/abc-123"
```

Interactive API docs available at `http://localhost:8000/docs`.

Project structure is documented in `src/README.md`.

### Embedding Providers

Paperfind supports multiple embedding providers for flexibility and local inference.

**OpenAI (default)**

```bash
# Uses OpenAI API (requires OPENAI_API_KEY)
export EMBEDDING_PROVIDER=openai
export EMBEDDING_MODEL=text-embedding-3-small  # default
```

**Ollama (local)**

Run embeddings locally using Ollama:

```bash
# Install the optional dependency
pip install paperfind[ollama]

# Configure
export EMBEDDING_PROVIDER=ollama
export EMBEDDING_MODEL=nomic-embed-text  # default
export OLLAMA_BASE_URL=http://localhost:11434  # optional

# Make sure Ollama is running and has the model
ollama pull nomic-embed-text
```

**HuggingFace (local)**

Run embeddings locally using sentence-transformers:

```bash
# Install the optional dependency
pip install paperfind[huggingface]

# Configure
export EMBEDDING_PROVIDER=huggingface
export EMBEDDING_MODEL=all-MiniLM-L6-v2  # default
```

**Switching Providers**

Each provider/model combination uses a separate vector store directory (e.g., `chroma_store_ollama_nomic-embed-text/`). When you switch providers or models, you need to rebuild your embeddings:

```bash
# After changing EMBEDDING_PROVIDER or EMBEDDING_MODEL
paperfind fetch --rebuild-vectors
paperfind embed "your collection" --force  # to rebuild Zotero vectors
```

## Data Storage

By default, data is stored in `~/.paperfind/` using SQLite and Chroma. To use Postgres,
install `paperfind[postgres]` and set `PAPERFIND_DB_URL` in your `.env`. Postgres uses one
database with two schemas (`daily`, `zotero`). To store embeddings in Postgres, set
`PAPERFIND_VECTOR_STORE=pgvector` (otherwise Chroma remains the default).

| File/Directory | Created By | Description |
|----------------|------------|-------------|
| `daily_papers.db` | `paperfind fetch` | SQLite database (default) of harvested papers from CrossRef, bioRxiv, medRxiv, arXiv |
| `zotero_meta.db` | `paperfind sync` | SQLite database (default) of your Zotero library (libraries, items, collections, tags) |
| `chroma_store_<provider>_<model>/` | `paperfind fetch --rebuild-vectors` | ChromaDB vector embeddings for daily papers |
| `zotero_vectors_<provider>_<model>/` | `paperfind embed` | ChromaDB vector embeddings for Zotero items |
| `.env` | Manual | Optional: API keys (can also be in current directory) |

### What Happens on Repeated Runs

| Command | Behavior |
|---------|----------|
| `paperfind sync` | **Upserts** all items from your entire Zotero library. Updates existing items, adds new ones, and refreshes collection memberships. Safe to run multiple times. |
| `paperfind embed <collection>` | **Skips** items already embedded. Only embeds new items in the collection. Use `--force` to re-embed all. |
| `paperfind fetch` | **Upserts** papers (updates existing records, adds new ones). Running daily accumulates papers over time. |
| `paperfind fetch --rebuild-vectors` | Fetches papers (upsert), then **recreates** the entire vector store from the database. |
| `paperfind fetch --vectors-only` | **Recreates** the vector store from existing database without fetching new papers. |
| `paperfind recommend` | Read-only. Queries existing databases. Creates output file only if `-o` specified. |
| `paperfind search` | Read-only. Queries existing vector stores. |

For database schema details, see [src/README.md](src/README.md#database-schemas)
