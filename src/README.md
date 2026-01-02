# Project Structure

```
src/
└── paperfind/            # Main package
    ├── __init__.py
    ├── cli.py            # Command-line interface
    ├── config.py         # Configuration and paths
    ├── embeddings.py     # Embedding provider support
    ├── rerank.py         # Cross-encoder reranking
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
