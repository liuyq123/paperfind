# Tests

Unit and integration tests for Paperfind.

## Setup

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest tests/test_cli.py  # Single file
pytest --cov=paperfind    # With coverage
```

## Test Files

| File | Description |
|------|-------------|
| `test_api_integration.py` | API endpoints (requires `[api]`) |
| `test_cli.py` | CLI commands and module imports |
| `test_config.py` | Configuration and paths |
| `test_digest_integration.py` | Digest workflow end-to-end |
| `test_digest_template.py` | HTML email template rendering |
| `test_documents.py` | Title/abstract extraction |
| `test_fetchers.py` | Paper source fetchers |
| `test_formatting.py` | Console and markdown formatting |
| `test_recommend.py` | Recommendation engine |
| `test_retry.py` | Retry logic for API calls |
| `test_search.py` | Semantic search |
| `test_sent_recommendations.py` | Sent DOI tracking |
| `test_vectorstore.py` | Vector store backends |

## Guidelines

- Mock all external API calls
- Use fixtures from `conftest.py`
- No real network calls
