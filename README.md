# Paperfind

A paper recommendation system that discovers relevant papers and preprints based on your Zotero library. It fetches metadata from CrossRef, bioRxiv, medRxiv, and arXiv, then uses semantic search to recommend papers similar to your existing research interests.

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
git clone <repository-url>
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

### Dependencies

The package will automatically install:
- requests
- python-dotenv
- langchain-core, langchain-openai, langchain-chroma
- chromadb
- openai

## Configuration

### Step 1: Create your `.env` file

Create a `.env` file in `~/.paperfind/` (recommended) or your current working directory:

```bash
# Create the data directory
mkdir -p ~/.paperfind

# Create the .env file
nano ~/.paperfind/.env
```

Add the following content:

```
OPENAI_API_KEY=sk-...
ZOTERO_API_KEY=...
ZOTERO_USER_ID=...
CROSSREF_EMAIL=your_email@example.com
SMTP_HOST=smtp.example.com
SMTP_PORT=587
SMTP_USER=your_smtp_user
SMTP_PASSWORD=your_smtp_password
EMAIL_FROM=paperfind@example.com
EMAIL_TO=you@example.com,teammate@example.com
```

`EMAIL_TO` accepts a comma-separated list of recipient addresses.

### Step 2: Verify configuration

```bash
paperfind config
```

This shows your data directory path. Your `.env` file should be in that directory or your current working directory.

### Custom data directory

To use a custom location for all data:

```bash
export PAPERFIND_DATA_DIR=/path/to/your/data
```

Add this to your shell profile (`.bashrc`, `.zshrc`, etc.) to make it permanent.

## Usage

### Quick Start: Get Today's Recommendations

```bash
# 1. Sync your Zotero library
paperfind sync

# 2. Fetch today's papers and build embeddings
paperfind fetch --rebuild-vectors

# 3. Get personalized recommendations based on your Zotero library
paperfind recommend
```

### Sync Zotero Library

Sync your Zotero library to get personalized recommendations:

```bash
# List available collections in your Zotero library
paperfind sync --list-collections

# Sync your entire library
paperfind sync

# Sync a specific collection
paperfind sync --collection "active learning"
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
```

The markdown file includes title, authors, abstract, date, source, and DOI links for each paper.

### Fetch Papers

Fetch papers from all sources with a single command:

```bash
# Fetch today's papers from all sources (CrossRef, bioRxiv, medRxiv, arXiv)
paperfind fetch

# Fetch last 7 days and rebuild vector embeddings
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

**Options:**

| Flag | Description |
|------|-------------|
| `-k`, `--num-results` | Number of results to return (default: 5) |
| `-s`, `--source` | Data source: `daily_papers` or `zotero` |
| `--rag` | Use RAG to answer the query as a question |
| `--scores` | Show similarity scores |
| `--project-id` | Filter by Zotero project ID |

## Project Structure

```
paperfind/
├── pyproject.toml            # Package configuration
├── README.md
├── src/
│   └── paperfind/            # Main package
│       ├── __init__.py
│       ├── cli.py            # Command-line interface
│       ├── config.py         # Configuration and paths
│       ├── fetchers/         # Paper fetching modules
│       │   ├── db.py
│       │   ├── fetch_papers.py
│       │   ├── vector.py
│       │   └── zotero/
│       │       ├── api.py
│       │       ├── db.py
│       │       ├── sync.py
│       │       └── vector.py
│       └── search/           # Search and recommendation
│           ├── recommend.py
│           └── search.py
└── notebooks/                # Jupyter notebooks (optional)
```

## Data Storage

By default, data is stored in `~/.paperfind/`:

| File/Directory | Created By | Description |
|----------------|------------|-------------|
| `daily_papers.db` | `paperfind fetch` | SQLite database of harvested papers from CrossRef, bioRxiv, medRxiv, arXiv |
| `zotero_meta.db` | `paperfind sync` | SQLite database of your Zotero library (projects, items, tags) |
| `chroma_store/` | `paperfind fetch --rebuild-vectors` | ChromaDB vector embeddings for daily papers |
| `zotero_vectors/` | `paperfind sync` | ChromaDB vector embeddings for Zotero items |
| `.env` | Manual | Optional: API keys (can also be in current directory) |

### What Happens on Repeated Runs

| Command | Behavior |
|---------|----------|
| `paperfind sync` | **Replaces** all items for the collection. Fetches fresh data from Zotero API and rebuilds vectors. Safe to run multiple times. |
| `paperfind fetch` | **Upserts** papers (updates existing records, adds new ones). Running daily accumulates papers over time. |
| `paperfind fetch --rebuild-vectors` | Fetches papers (upsert), then **recreates** the entire vector store from the database. |
| `paperfind fetch --vectors-only` | **Recreates** the vector store from existing database without fetching new papers. |
| `paperfind recommend` | Read-only. Queries existing databases. Creates output file only if `-o` specified. |
| `paperfind search` | Read-only. Queries existing vector stores. |

### Database Schemas

**daily_papers.db** (table: `works`)
- `doi` (PRIMARY KEY) - Paper identifier
- `title`, `authors`, `abstract` - Paper metadata
- `created_date` - Publication date
- `type` - Article type (journal-article, preprint, etc.)
- `source` - Where it was fetched from (crossref, biorxiv, arxiv, etc.)

**zotero_meta.db** (tables: `projects`, `items`, `tags`)
- `projects` - Zotero collections/libraries synced
- `items` - Papers with metadata (zotero_key, title, authors, abstract, DOI, date, URL)
- `tags` - Research tags for organization

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key for embeddings and LLM |
| `ZOTERO_API_KEY` | Zotero API key for library sync |
| `ZOTERO_USER_ID` | Your Zotero user ID |
| `CROSSREF_EMAIL` | Email for CrossRef API (polite pool) |
| `SMTP_HOST` | SMTP server hostname |
| `SMTP_PORT` | SMTP server port |
| `SMTP_USER` | SMTP username |
| `SMTP_PASSWORD` | SMTP password |
| `EMAIL_FROM` | From address for digest emails |
| `EMAIL_TO` | Comma-separated list of digest recipients |
| `PAPERFIND_DATA_DIR` | Custom data directory (optional) |
| `EMBEDDING_PROVIDER` | Embedding provider: `openai`, `ollama`, or `huggingface` (default: `openai`) |
| `EMBEDDING_MODEL` | Model name (provider-specific defaults apply) |
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://localhost:11434`) |

## Embedding Providers

Paperfind supports multiple embedding providers for flexibility and local inference.

### OpenAI (default)

```bash
# Uses OpenAI API (requires OPENAI_API_KEY)
export EMBEDDING_PROVIDER=openai
export EMBEDDING_MODEL=text-embedding-3-small  # default
```

### Ollama (local)

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

### HuggingFace (local)

Run embeddings locally using sentence-transformers:

```bash
# Install the optional dependency
pip install paperfind[huggingface]

# Configure
export EMBEDDING_PROVIDER=huggingface
export EMBEDDING_MODEL=all-MiniLM-L6-v2  # default
```

### Switching Providers

Each provider/model combination uses a separate vector store directory (e.g., `chroma_store_ollama_nomic-embed-text/`). When you switch providers or models, you need to rebuild your embeddings:

```bash
# After changing EMBEDDING_PROVIDER or EMBEDDING_MODEL
paperfind fetch --rebuild-vectors
paperfind sync  # to rebuild Zotero vectors
```
