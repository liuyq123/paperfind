"""
Configuration and path management for Paperfind.

Data is stored in ~/.paperfind/ by default, or in PAPERFIND_DATA_DIR if set.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


def get_data_dir() -> Path:
    """Get the data directory, creating it if needed."""
    if os.getenv("PAPERFIND_DATA_DIR"):
        data_dir = Path(os.getenv("PAPERFIND_DATA_DIR"))
    else:
        data_dir = Path.home() / ".paperfind"

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def load_config():
    """Load configuration from .env file."""
    # Try loading from current directory first
    if Path(".env").exists():
        load_dotenv(".env")

    # Then try data directory
    data_dir = get_data_dir()
    env_file = data_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)


# Initialize on import
load_config()

# Paths
DATA_DIR = get_data_dir()
DAILY_PAPERS_DB = DATA_DIR / "daily_papers.db"
ZOTERO_DB = DATA_DIR / "zotero_meta.db"


def _get_model_suffix() -> str:
    """Get sanitized provider and model name for directory suffix."""
    from paperfind.embeddings import get_embedding_model, get_embedding_provider, sanitize_model_name

    provider = get_embedding_provider()
    model = sanitize_model_name(get_embedding_model())
    return f"{provider}_{model}"


def get_chroma_store_dir() -> str:
    """Get the ChromaDB store directory for the current model."""
    return str(DATA_DIR / f"chroma_store_{_get_model_suffix()}")


def get_zotero_vectors_dir() -> str:
    """Get the Zotero vectors directory for the current model."""
    return str(DATA_DIR / f"zotero_vectors_{_get_model_suffix()}")


# API Keys and settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
ZOTERO_LIBRARY_TYPE = os.getenv("ZOTERO_LIBRARY_TYPE", "user")
CROSSREF_EMAIL = os.getenv("CROSSREF_EMAIL")

# Model settings
# Note: EMBEDDING_PROVIDER and EMBEDDING_MODEL are handled by paperfind.embeddings
# with provider-specific defaults. Use get_embedding_provider() and get_embedding_model().
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# Paper source categories (configurable via comma-separated env vars)
_DEFAULT_ARXIV_CATEGORIES = [
    "q-bio.BM",        # Biomolecules
    "q-bio.QM",        # Quantitative Methods
    "q-bio.MN",        # Molecular Networks
    "cs.LG",           # Machine Learning
    "cs.AI",           # Artificial Intelligence
    "stat.ML",         # Machine Learning (Statistics)
    "physics.chem-ph", # Chemical Physics
    "physics.bio-ph",  # Biological Physics
]
_DEFAULT_BIORXIV_CATEGORIES = [
    "bioinformatics",
    "biochemistry",
    "pharmacology-and-toxicology",
    "systems-biology",
    "synthetic-biology",
    "molecular-biology",
    "cell-biology",
    "genomics",
    "biophysics",
]


def _parse_categories(env_var: str, defaults: list) -> list:
    """Parse comma-separated category list from env var."""
    value = os.getenv(env_var)
    if value:
        return [c.strip() for c in value.split(",") if c.strip()]
    return defaults


ARXIV_CATEGORIES = _parse_categories("ARXIV_CATEGORIES", _DEFAULT_ARXIV_CATEGORIES)
BIORXIV_CATEGORIES = _parse_categories("BIORXIV_CATEGORIES", _DEFAULT_BIORXIV_CATEGORIES)

# Email settings for digest
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO = os.getenv("EMAIL_TO")
