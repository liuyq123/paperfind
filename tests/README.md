# Tests

This folder contains unit and light integration tests for Paperfind.

## Setup

Install dev dependencies first:

```bash
pip install -e ".[dev]"
```

## Test Files

| File | Description | Requires API? |
|------|-------------|---------------|
| `test_documents.py` | Metadata-first title/abstract extraction and fallbacks | No |
| `test_formatting.py` | Console/markdown formatting uses metadata fields | No |
| `test_digest_template.py` | HTML digest uses metadata and link formatting | No |
| `test_api_integration.py` | API handlers with mocked HTTP backends | No |

## How to Run

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a single file
pytest tests/test_documents.py

# Run a single test
pytest tests/test_documents.py::test_extract_title

# Run with coverage
pytest --cov=paperfind --cov-report=term-missing
```

## Writing Tests

- All external API calls should be mocked/monkeypatched
- Use fixtures from `conftest.py` where available
- Keep tests fast - no real network calls
