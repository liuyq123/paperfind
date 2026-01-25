"""
Configuration and path management for Paperfind.

Data is stored in ~/.paperfind/ by default, or in PAPERFIND_DATA_DIR if set.

Config loading priority:
1. Explicit path via load_config(config_path)
2. PAPERFIND_CONFIG environment variable
3. .env file in current directory
4. ~/.paperfind/.env (data directory)
5. No file (uses existing env vars) - for CI/GitHub Actions
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Track whether config has been loaded
_config_loaded = False
_config_path: Optional[Path] = None


def get_data_dir() -> Path:
    """Get the data directory, creating it if needed.
    Returns PAPERFIND_DATA_DIR if set, otherwise ~/.paperfind.
    """
    if os.getenv("PAPERFIND_DATA_DIR"):
        data_dir = Path(os.getenv("PAPERFIND_DATA_DIR"))
    else:
        data_dir = Path.home() / ".paperfind"

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def load_config(config_path: Optional[str] = None) -> Optional[Path]:
    """Load configuration from .env file or use existing environment variables.

    Args:
        config_path: Explicit path to .env file. If not provided, checks
                     PAPERFIND_CONFIG env var, then local .env, then
                     ~/.paperfind/.env, then falls back to using existing
                     environment variables (for CI).

    Returns:
        Path to the loaded .env file, or None if using environment variables.

    Raises:
        FileNotFoundError: If explicit config_path doesn't exist.
    """
    global _config_loaded, _config_path

    env_file: Optional[Path] = None

    # Priority 1: Explicit path
    if config_path:
        env_file = Path(config_path)
        if not env_file.exists():
            raise FileNotFoundError(f"Config file not found: {env_file}")

    # Priority 2: PAPERFIND_CONFIG env var
    elif os.getenv("PAPERFIND_CONFIG"):
        env_file = Path(os.getenv("PAPERFIND_CONFIG"))
        if not env_file.exists():
            raise FileNotFoundError(
                f"Config file not found: {env_file} (from PAPERFIND_CONFIG)"
            )

    # Priority 3: Local .env (current directory)
    elif Path(".env").exists():
        env_file = Path(".env")

    # Priority 4: Data directory .env (~/.paperfind/.env)
    elif (Path.home() / ".paperfind" / ".env").exists():
        env_file = Path.home() / ".paperfind" / ".env"

    # Priority 5: No file - use existing environment variables (CI/GitHub Actions)
    # This is fine, env vars should already be set

    if env_file:
        load_dotenv(env_file, override=True)

    _config_loaded = True
    _config_path = env_file

    # Reload module-level variables after loading config
    _reload_config_values()

    return env_file


def _reload_config_values():
    """Reload all module-level config values from environment."""
    global DATA_DIR, DAILY_PAPERS_DB, ZOTERO_DB
    global OPENAI_API_KEY, ZOTERO_API_KEY, ZOTERO_USER_ID, ZOTERO_LIBRARY_TYPE
    global CROSSREF_EMAIL, LLM_MODEL, LLM_RERANK_MODEL
    global ARXIV_CATEGORIES, BIORXIV_CATEGORIES, MEDRXIV_CATEGORIES, CHEMRXIV_CATEGORIES
    global SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, EMAIL_FROM, EMAIL_TO

    # Paths
    DATA_DIR = get_data_dir()
    DAILY_PAPERS_DB = DATA_DIR / "daily_papers.db"
    ZOTERO_DB = DATA_DIR / "zotero_meta.db"

    # API Keys and settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ZOTERO_API_KEY = os.getenv("ZOTERO_API_KEY")
    ZOTERO_USER_ID = os.getenv("ZOTERO_USER_ID")
    ZOTERO_LIBRARY_TYPE = os.getenv("ZOTERO_LIBRARY_TYPE", "user")
    CROSSREF_EMAIL = os.getenv("CROSSREF_EMAIL")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_RERANK_MODEL = os.getenv("LLM_RERANK_MODEL", "gpt-4o-mini")

    # Paper source categories
    ARXIV_CATEGORIES = _parse_categories(
        "ARXIV_CATEGORIES", _DEFAULT_ARXIV_CATEGORIES, empty_fallback=_ALL_ARXIV_CATEGORIES
    )
    BIORXIV_CATEGORIES = _parse_categories(
        "BIORXIV_CATEGORIES", _DEFAULT_BIORXIV_CATEGORIES, empty_fallback=[]
    )
    MEDRXIV_CATEGORIES = _parse_categories(
        "MEDRXIV_CATEGORIES", _DEFAULT_MEDRXIV_CATEGORIES, empty_fallback=[]
    )
    CHEMRXIV_CATEGORIES = _parse_categories(
        "CHEMRXIV_CATEGORIES", _DEFAULT_CHEMRXIV_CATEGORIES, empty_fallback=[]
    )

    # Email settings
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT") or "587")
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
    EMAIL_FROM = os.getenv("EMAIL_FROM")
    EMAIL_TO = os.getenv("EMAIL_TO")


def is_config_loaded() -> bool:
    """Check if configuration has been loaded."""
    return _config_loaded


def get_loaded_config_path() -> Optional[Path]:
    """Get the path to the loaded config file, or None if not loaded."""
    return _config_path


# =============================================================================
# Default category lists (used by _parse_categories)
# =============================================================================

_DEFAULT_ARXIV_CATEGORIES = [
    # Quantitative Biology (all subcategories)
    "q-bio.BM",        # Biomolecules
    "q-bio.CB",        # Cell Behavior
    "q-bio.GN",        # Genomics
    "q-bio.MN",        # Molecular Networks
    "q-bio.NC",        # Neurons and Cognition
    "q-bio.OT",        # Other Quantitative Biology
    "q-bio.PE",        # Populations and Evolution
    "q-bio.QM",        # Quantitative Methods
    "q-bio.SC",        # Subcellular Processes
    "q-bio.TO",        # Tissues and Organs
    # AI/ML categories
    "cs.AI",           # Artificial Intelligence
    "cs.CL",           # Computation and Language (NLP)
    "cs.CV",           # Computer Vision
    "cs.IR",           # Information Retrieval
    "cs.LG",           # Machine Learning
    "cs.NE",           # Neural and Evolutionary Computing
    "cs.RO",           # Robotics
    "stat.ML",         # Machine Learning (Statistics)
    # Physics
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
    "genetics",
    "biophysics",
]

_DEFAULT_MEDRXIV_CATEGORIES: list[str] = [
    "genetic-and-genomic-medicine",
    "health-informatics",
]

_DEFAULT_CHEMRXIV_CATEGORIES = [
    "605c72ef153207001f6470d0",  # Biological and Medicinal Chemistry
    "605c72ef153207001f6470ce",  # Theoretical and Computational Chemistry
]


def _parse_categories(
    env_var: str, defaults: list, empty_fallback: list | None = None
) -> list:
    """Parse comma-separated category list from env var."""
    value = os.getenv(env_var)
    if value is None:
        return defaults
    if value.strip() == "":
        return empty_fallback if empty_fallback is not None else defaults
    return [c.strip() for c in value.split(",") if c.strip()]


# =============================================================================
# Module-level config variables (initialized to None, set by load_config)
# =============================================================================

# Paths
DATA_DIR: Path = Path.home() / ".paperfind"  # Default, updated by load_config
DAILY_PAPERS_DB: Path = DATA_DIR / "daily_papers.db"
ZOTERO_DB: Path = DATA_DIR / "zotero_meta.db"

# API Keys and settings
OPENAI_API_KEY: Optional[str] = None
ZOTERO_API_KEY: Optional[str] = None
ZOTERO_USER_ID: Optional[str] = None
ZOTERO_LIBRARY_TYPE: str = "user"
CROSSREF_EMAIL: Optional[str] = None
LLM_MODEL: str = "gpt-4o-mini"
LLM_RERANK_MODEL: str = "gpt-4o-mini"

# Paper source categories
ARXIV_CATEGORIES: list[str] = _DEFAULT_ARXIV_CATEGORIES
BIORXIV_CATEGORIES: list[str] = _DEFAULT_BIORXIV_CATEGORIES
MEDRXIV_CATEGORIES: list[str] = _DEFAULT_MEDRXIV_CATEGORIES
CHEMRXIV_CATEGORIES: list[str] = _DEFAULT_CHEMRXIV_CATEGORIES

# Email settings
SMTP_HOST: str = "smtp.gmail.com"
SMTP_PORT: int = 587
SMTP_USER: Optional[str] = None
SMTP_PASSWORD: Optional[str] = None
EMAIL_FROM: Optional[str] = None
EMAIL_TO: Optional[str] = None


# =============================================================================
# Helper functions
# =============================================================================

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


def get_rerank_preferences() -> Optional[str]:
    """Load user preferences for LLM-based reranking.

    Checks in order:
    1. LLM_RERANK_PREFERENCES environment variable
    2. rerank_preferences.txt file in data directory

    Returns:
        User preferences as a string, or None if not configured.
    """
    # Priority 1: Environment variable
    if prefs := os.getenv("LLM_RERANK_PREFERENCES"):
        return prefs

    # Priority 2: Preferences file
    prefs_file = DATA_DIR / "rerank_preferences.txt"
    if prefs_file.exists():
        return prefs_file.read_text().strip()

    return None


def get_keywords() -> Optional[list[str]]:
    """Load default keywords from environment variable.

    Reads from PAPERFIND_KEYWORDS env var. Keywords should be separated by
    semicolons (;) to allow multi-word phrases.

    Example:
        PAPERFIND_KEYWORDS="protein design;drug discovery;machine learning"

    Returns:
        List of keyword phrases, or None if not configured.
    """
    keywords_str = os.getenv("PAPERFIND_KEYWORDS")
    if not keywords_str:
        return None

    # Split by semicolon and strip whitespace
    keywords = [k.strip() for k in keywords_str.split(";") if k.strip()]
    return keywords if keywords else None


# =============================================================================
# Configuration validation
# =============================================================================
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
        - config_file: Path to loaded config file (or None)
        - config_loaded: Whether config has been loaded
        - operations: Dict of operation -> list of missing vars
    """
    from paperfind.embeddings import get_embedding_model, get_embedding_provider

    operations = {
        "zotero": validate_config("zotero", raise_on_error=False),
        "embeddings": validate_config("embeddings", raise_on_error=False),
        "email": validate_config("email", raise_on_error=False),
    }

    return {
        "data_dir": str(DATA_DIR),
        "config_file": str(_config_path) if _config_path else None,
        "config_loaded": _config_loaded,
        "embedding_provider": get_embedding_provider(),
        "embedding_model": get_embedding_model(),
        "operations": operations,
    }
