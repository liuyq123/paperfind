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

# Full list of arXiv categories (used when ARXIV_CATEGORIES is set to empty)
# See https://arxiv.org/category_taxonomy
_ALL_ARXIV_CATEGORIES = [
    # Computer Science
    "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV",
    "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL",
    "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO",
    "cs.MA", "cs.MM", "cs.MS", "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS",
    "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY",
    # Economics
    "econ.EM", "econ.GN", "econ.TH",
    # Electrical Engineering and Systems Science
    "eess.AS", "eess.IV", "eess.SP", "eess.SY",
    # Mathematics
    "math.AC", "math.AG", "math.AP", "math.AT", "math.CA", "math.CO", "math.CT",
    "math.CV", "math.DG", "math.DS", "math.FA", "math.GM", "math.GN", "math.GR",
    "math.GT", "math.HO", "math.IT", "math.KT", "math.LO", "math.MG", "math.MP",
    "math.NA", "math.NT", "math.OA", "math.OC", "math.PR", "math.QA", "math.RA",
    "math.RT", "math.SG", "math.SP", "math.ST",
    # Physics
    "astro-ph.CO", "astro-ph.EP", "astro-ph.GA", "astro-ph.HE", "astro-ph.IM",
    "astro-ph.SR", "cond-mat.dis-nn", "cond-mat.mes-hall", "cond-mat.mtrl-sci",
    "cond-mat.other", "cond-mat.quant-gas", "cond-mat.soft", "cond-mat.stat-mech",
    "cond-mat.str-el", "cond-mat.supr-con", "gr-qc", "hep-ex", "hep-lat", "hep-ph",
    "hep-th", "math-ph", "nlin.AO", "nlin.CD", "nlin.CG", "nlin.PS", "nlin.SI",
    "nucl-ex", "nucl-th", "physics.acc-ph", "physics.ao-ph", "physics.app-ph",
    "physics.atm-clus", "physics.atom-ph", "physics.bio-ph", "physics.chem-ph",
    "physics.class-ph", "physics.comp-ph", "physics.data-an", "physics.ed-ph",
    "physics.flu-dyn", "physics.gen-ph", "physics.geo-ph", "physics.hist-ph",
    "physics.ins-det", "physics.med-ph", "physics.optics", "physics.plasm-ph",
    "physics.pop-ph", "physics.soc-ph", "physics.space-ph", "quant-ph",
    # Quantitative Biology
    "q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC", "q-bio.OT",
    "q-bio.PE", "q-bio.QM", "q-bio.SC", "q-bio.TO",
    # Quantitative Finance
    "q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM", "q-fin.PR",
    "q-fin.RM", "q-fin.ST", "q-fin.TR",
    # Statistics
    "stat.AP", "stat.CO", "stat.ME", "stat.ML", "stat.OT", "stat.TH",
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
_DEFAULT_MEDRXIV_CATEGORIES: list[str] = []


def _parse_categories(
    env_var: str, defaults: list, empty_fallback: list | None = None
) -> list:
    """Parse comma-separated category list from env var.

    Args:
        env_var: Name of the environment variable
        defaults: Default categories if env var is not set
        empty_fallback: Categories to use when env var is explicitly set to empty.
                        If None, returns defaults. If empty list [], returns [].
    """
    value = os.getenv(env_var)
    if value is None:
        return defaults
    if value.strip() == "":
        # Explicitly set to empty
        return empty_fallback if empty_fallback is not None else defaults
    return [c.strip() for c in value.split(",") if c.strip()]


# arXiv: empty = fetch all categories (slow, ~100 categories)
ARXIV_CATEGORIES = _parse_categories(
    "ARXIV_CATEGORIES", _DEFAULT_ARXIV_CATEGORIES, empty_fallback=_ALL_ARXIV_CATEGORIES
)
# bioRxiv: empty = fetch all (no category filtering)
BIORXIV_CATEGORIES = _parse_categories(
    "BIORXIV_CATEGORIES", _DEFAULT_BIORXIV_CATEGORIES, empty_fallback=[]
)
# medRxiv: empty/default = fetch all (no category filtering)
MEDRXIV_CATEGORIES = _parse_categories(
    "MEDRXIV_CATEGORIES", _DEFAULT_MEDRXIV_CATEGORIES, empty_fallback=[]
)

# Email settings for digest
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT") or "587")
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM")
EMAIL_TO = os.getenv("EMAIL_TO")


# Configuration validation
class ConfigValidationError(Exception):
    """Raised when required configuration is missing."""

    def __init__(self, missing: list[str], context: str = ""):
        self.missing = missing
        self.context = context
        msg = f"Missing required configuration: {', '.join(missing)}"
        if context:
            msg = f"{context}: {msg}"
        super().__init__(msg)


def validate_config(
    operation: str = "general",
    raise_on_error: bool = True,
) -> list[str]:
    """Validate configuration for a specific operation.

    Args:
        operation: The operation to validate for. Options:
            - "zotero": Requires ZOTERO_API_KEY and ZOTERO_USER_ID
            - "embeddings": Requires OPENAI_API_KEY (for OpenAI provider)
            - "email": Requires SMTP_USER, SMTP_PASSWORD, EMAIL_FROM, EMAIL_TO
            - "general": Basic validation (no specific requirements)
        raise_on_error: If True, raise ConfigValidationError on missing config

    Returns:
        List of missing configuration variable names (empty if valid)

    Raises:
        ConfigValidationError: If raise_on_error is True and config is invalid
    """
    from paperfind.embeddings import get_embedding_provider

    missing: list[str] = []

    if operation == "zotero":
        if not ZOTERO_API_KEY:
            missing.append("ZOTERO_API_KEY")
        if not ZOTERO_USER_ID:
            missing.append("ZOTERO_USER_ID")

    elif operation == "embeddings":
        provider = get_embedding_provider()
        if provider == "openai" and not OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        # Note: ollama and huggingface don't require API keys

    elif operation == "email":
        if not SMTP_USER:
            missing.append("SMTP_USER")
        if not SMTP_PASSWORD:
            missing.append("SMTP_PASSWORD")
        if not EMAIL_FROM:
            missing.append("EMAIL_FROM")
        if not EMAIL_TO:
            missing.append("EMAIL_TO")

    if missing and raise_on_error:
        raise ConfigValidationError(missing, operation)

    return missing


def check_config(operation: str = "general") -> bool:
    """Check if configuration is valid for an operation.

    Returns True if valid, False if missing required configuration.
    This is a convenience wrapper around validate_config that doesn't raise.
    """
    return len(validate_config(operation, raise_on_error=False)) == 0


def get_config_status() -> dict:
    """Get the current configuration status.

    Returns a dict with:
        - data_dir: Path to data directory
        - env_file_loaded: Whether .env was found
        - operations: Dict of operation -> list of missing vars
    """
    from paperfind.embeddings import get_embedding_model, get_embedding_provider

    # Check if .env file exists
    env_file_loaded = False
    if Path(".env").exists():
        env_file_loaded = True
    elif (DATA_DIR / ".env").exists():
        env_file_loaded = True

    operations = {
        "zotero": validate_config("zotero", raise_on_error=False),
        "embeddings": validate_config("embeddings", raise_on_error=False),
        "email": validate_config("email", raise_on_error=False),
    }

    return {
        "data_dir": str(DATA_DIR),
        "env_file_loaded": env_file_loaded,
        "embedding_provider": get_embedding_provider(),
        "embedding_model": get_embedding_model(),
        "operations": operations,
    }
