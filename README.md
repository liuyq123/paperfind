# Paperfind

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A paper recommendation system that discovers relevant papers based on your Zotero library. Fetches from CrossRef, bioRxiv, medRxiv, arXiv, and ChemRxiv, then uses semantic search to recommend papers matching your research interests.

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
  - [Generate Rerank Preferences](#generate-rerank-preferences)
  - [Improve Preferences with Feedback](#improve-preferences-with-feedback)
  - [Semantic Search](#semantic-search)
  - [Prune Old Papers](#prune-old-papers)
  - [Embedding Providers](#embedding-providers)
  - [API Server (optional)](#api-server-optional)
- [Data Storage](#data-storage)
- [Internals](#internals)
- [License](#license)

## Features

- **Recommendations**: Discover papers similar to your Zotero library
- **Semantic Keyword Matching**: Find papers by topic using natural language keywords (e.g., "protein design" matches related concepts)
- **LLM Reranking**: Use GPT-4o-mini to rerank papers based on your custom research preferences
- **Multi-source Fetching**: CrossRef, bioRxiv, medRxiv, arXiv, ChemRxiv
- **Semantic Search**: Vector search with OpenAI, Ollama, or HuggingFace embeddings
- **RAG**: Ask questions about your paper collection
- **Email Digest**: Scheduled recommendations via GitHub Actions

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
```

**Semantic keyword matching**

Use `--keywords` to find papers by topic. Keywords are matched semantically - "protein design" will find papers about related concepts like "de novo protein synthesis" or "computational protein folding" even if they don't contain the exact phrase:

```bash
# Find papers about a specific topic
paperfind recommend --keywords "protein design"

# Combine multiple keywords (results are merged)
paperfind recommend --keywords "protein design" "machine learning"

# Combine keywords with Zotero-based recommendations
# Papers matching either source are included
paperfind recommend --collection "my research" --keywords "drug discovery"
```

The markdown file includes title, authors, abstract, date, source, and DOI links for each paper.

**LLM-based reranking with user preferences**

For more personalized results, use `--rerank` to have an LLM (GPT-4o-mini by default) score papers based on your custom research preferences:

```bash
# First, generate a preferences file
paperfind init-preferences --keywords "protein design" "drug discovery"

# Edit the generated file at ~/.paperfind/rerank_preferences.txt
# Then use LLM reranking
paperfind recommend --rerank
```

The preferences file lets you specify what you're interested in and what you're NOT interested in. See [`rerank_preferences.example.txt`](rerank_preferences.example.txt) for an example.

**Default keywords via environment variable**

Instead of passing `--keywords` every time, you can set default keywords in your `.env` file:

```bash
# In .env (semicolon-separated for multi-word phrases)
PAPERFIND_KEYWORDS="protein design;drug discovery;machine learning"
```

CLI `--keywords` flags override this setting when provided.

### Fetch Papers

Fetch papers from all sources with a single command:

```bash
# Fetch recent papers from all sources (CrossRef, bioRxiv, medRxiv, arXiv, ChemRxiv)
paperfind fetch

# Fetch last 7 days and rebuild vector embeddings
paperfind fetch --days 7 --rebuild-vectors

# Fetch from specific sources only
paperfind fetch --source arxiv biorxiv

# Only rebuild vectors (no fetching)
paperfind fetch --vectors-only
```

**Note on `--days`:** The default is 2 days to handle timezone differences between your machine and the API servers. Since papers are upserted, fetching duplicates is harmless.

**Sources and categories:**
- **CrossRef**: Journal articles and preprints with DOIs
- **bioRxiv**: Life science preprints. Categories configured via `BIORXIV_CATEGORIES` env var.
- **medRxiv**: Clinical preprints. Categories configured via `MEDRXIV_CATEGORIES` env var.
- **arXiv**: Preprints. Categories configured via `ARXIV_CATEGORIES` env var.
- **ChemRxiv**: Chemistry preprints.

See [`.env.example`](.env.example) for default categories and customization.

**Note on arXiv delays:** arXiv has a delay between the publish date and when papers become available via the API. Use `--arxiv-days` to fetch a longer window from arXiv while keeping a shorter window for other sources.

### Email Digest

Send a scheduled email with the latest recommendations:

```bash
# Send today's digest email
paperfind digest

# Preview the email without sending
paperfind digest --dry-run

# Include the last 7 days of papers in the digest
paperfind digest --days 7

# Fetch 1 day from most sources, but 7 days from arXiv
paperfind digest --days 1 --arxiv-days 7

# Include more recommendations in the email
paperfind digest -k 20

# Skip fetching new papers before generating the digest
paperfind digest --skip-fetch

# Resend papers from the last digest (if you deleted the email)
paperfind digest --include-last-digests 1

# Include papers matching specific keywords
paperfind digest --keywords "protein design"
paperfind digest --keywords "drug discovery" "machine learning"

# Use LLM reranking for better personalization
paperfind digest --rerank
```

**Required SMTP settings**

Email delivery requires SMTP configuration. See [`.env.example`](.env.example) for the required variables (`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `EMAIL_FROM`, `EMAIL_TO`).

**Note for Gmail users:** Use an [App Password](https://myaccount.google.com/apppasswords) instead of your regular password if you have 2-factor authentication enabled.

**Avoiding repeat recommendations**

The digest automatically tracks which papers have been sent and excludes them from future recommendations. Sent papers are recorded after each successful email and expire after 30 days, allowing them to resurface if still relevant.

If you accidentally delete a digest email, use `--include-last-digests N` to include papers from the last N digests in your next email.

**Scheduled runs with GitHub Actions**

To run the digest on a schedule, see [`.github/workflows/digest.yml`](.github/workflows/digest.yml). Store your credentials as repository secrets (Settings → Secrets and variables → Actions):

- `PAPERFIND_KEYWORDS`: Default keywords (semicolon-separated, e.g., `protein design;drug discovery`)
- `LLM_RERANK_PREFERENCES`: Your preferences text for LLM reranking (optional)
- Plus SMTP credentials, API keys, etc.

### Generate Rerank Preferences

Generate a preferences file for LLM-based reranking. The command uses an LLM to create an initial preferences file based on your keywords, Zotero collection, or library contents:

```bash
# Generate from keywords
paperfind init-preferences --keywords "protein design" "drug discovery"

# Generate from a Zotero collection name + keywords
paperfind init-preferences --collection "active learning" --keywords "bayesian optimization"

# Generate from your Zotero library (uses paper titles as context)
paperfind init-preferences

# Save to a custom location
paperfind init-preferences --keywords "ML" -o ~/my_preferences.txt
```

The generated file is saved to `~/.paperfind/rerank_preferences.txt` by default. Edit it to refine your interests, then use `paperfind recommend --rerank` or `paperfind digest --rerank`.

**Preferences file format:**

```text
I'm interested in:
- Machine learning for drug discovery
- Protein structure prediction
- Active learning for molecular property prediction

I'm NOT interested in:
- Language models for law or finance
- Pure computer vision without biology applications

Notes:
- Papers added for datasets don't represent my core interests
```

### Improve Preferences with Feedback

After receiving recommendations, you can provide feedback to improve future results. Feedback immediately updates your preferences file:

```bash
# Add a positive preference (you want more of this type)
paperfind feedback --like -r "Virtual screening methodology and failure analysis papers"

# Add a negative preference (you want to avoid this type)
paperfind feedback --dislike -r "General antiviral research - only interested in screening methodology"

# Apply changes without confirmation prompt
paperfind feedback --like -r "Docking accuracy studies" --yes
```

The `-r/--reason` flag describes the TYPE of paper you want more or fewer of. You don't need to reference specific papers—just describe what you're looking for or want to avoid.

**Example workflow:**

```bash
# You receive a digest with some irrelevant papers about antivirals
# (you added a paper to study screening failures, not antivirals)
paperfind feedback --dislike \
  -r "General antiviral/antimicrobial research - only interested in screening methodology"

# You want more papers like the good ones you received
paperfind feedback --like \
  -r "Papers analyzing why virtual screening predictions fail"

# Output shows current vs proposed preferences for confirmation:
# CURRENT PREFERENCES:
# I'm interested in: virtual screening, docking...
#
# PROPOSED UPDATED PREFERENCES:
# I'm interested in:
# - Virtual screening methodology and failure analysis
# - Papers analyzing prediction accuracy
# ...
# I'm NOT interested in:
# - General antiviral research (unless about screening methods)
# ...
#
# Apply these changes? [y/N]
```

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

### Prune Old Papers

Over time, the database and vector store grow as you fetch papers daily. Use the prune command to delete old papers and keep storage manageable:

```bash
# Preview what would be deleted (dry run)
paperfind prune --older-than 30 --dry-run

# Delete papers older than 30 days from database and vector store
paperfind prune --older-than 30

# See individual DOIs in dry run (with verbose flag)
paperfind prune --older-than 30 --dry-run -v
```

The prune command:
- Deletes papers from the database where `created_date` is older than the specified number of days
- Removes corresponding embeddings from the vector store
- Does not affect your Zotero library (only daily papers)

**Automated pruning:** The GitHub Actions workflow ([`.github/workflows/digest.yml`](.github/workflows/digest.yml)) automatically prunes papers older than 30 days after each digest run.

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

### API Server (optional)

For programmatic access, Paperfind includes a REST API. Install with `pip install paperfind[api]` and see [src/README.md](src/README.md#api-server) for endpoints and examples.

## Data Storage

By default, data is stored in `~/.paperfind/` using SQLite and Chroma. To use Postgres,
install `paperfind[postgres]` and set `PAPERFIND_DB_URL` in your `.env`. Postgres uses one
database with two schemas (`daily`, `zotero`). To store embeddings in Postgres, set
`PAPERFIND_VECTOR_STORE=pgvector` (otherwise Chroma remains the default).

| File/Directory | Description | Commands |
|----------------|-------------|----------|
| `daily_papers.db` | SQLite database of harvested papers | `fetch` upserts; `prune` deletes old; `digest` tracks sent DOIs |
| `zotero_meta.db` | SQLite database of your Zotero library | `sync` upserts all items and collections |
| `chroma_store_<provider>_<model>/` | Vector embeddings for daily papers | `fetch --rebuild-vectors` recreates; `prune` removes old |
| `zotero_vectors_<provider>_<model>/` | Vector embeddings for Zotero items | `embed` adds new (use `--force` to re-embed all) |
| `.env` | API keys and configuration | Manual |

For database schema details, see [src/README.md](src/README.md#database-schemas).

## Internals

Project internals and architecture live in [src/README.md](src/README.md).
Test setup and commands live in [tests/README.md](tests/README.md).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
