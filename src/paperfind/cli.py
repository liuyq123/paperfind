"""
Command-line interface for Paperfind.

Usage:
    paperfind sync                     # Sync entire Zotero library
    paperfind embed                    # Embed all items in library
    paperfind embed "collection"       # Embed items in a collection
    paperfind fetch                    # Fetch papers from all sources
    paperfind recommend                # Get paper recommendations
    paperfind search "query"           # Semantic search
    paperfind digest                   # Send email digest of recommendations
    paperfind prune --older-than 30    # Delete papers older than 30 days
"""

import argparse
import sys
from typing import Optional


def positive_int(value: str) -> int:
    """Parse a positive integer from CLI input."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer value: {value}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("Value must be >= 1")
    return parsed


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="paperfind",
        description="Paper recommendation system based on your Zotero library",
    )

    # Global options
    parser.add_argument(
        "-c", "--config",
        type=str,
        metavar="PATH",
        help="Path to .env config file (default: .env in current directory)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output with timestamps",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set log level (default: INFO, or PAPERFIND_LOG_LEVEL env var)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Sync command (always syncs entire library)
    sync_parser = subparsers.add_parser("sync", help="Sync Zotero library to local database")
    sync_parser.add_argument(
        "--list-collections", action="store_true", help="List all collections in your library"
    )

    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Embed Zotero items for semantic search")
    embed_parser.add_argument(
        "collection", type=str, nargs="?", default=None,
        help="Collection name or key (omit to embed all items)"
    )
    embed_parser.add_argument(
        "--force", action="store_true", help="Re-embed all items (ignore existing embeddings)"
    )

    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch papers from external sources")
    fetch_parser.add_argument(
        "--days", type=positive_int, default=2,
        help="Number of days to look back (default: 2, to handle timezone differences)"
    )
    fetch_parser.add_argument(
        "--arxiv-days",
        type=positive_int,
        default=None,
        help="Number of days to look back for arXiv (default: same as --days). "
             "Useful since arXiv has batch processing delays.",
    )
    fetch_parser.add_argument(
        "--source",
        action="append",
        choices=["crossref", "biorxiv", "medrxiv", "arxiv", "chemrxiv"],
        help="Source(s) to fetch from (default: all)",
    )
    fetch_parser.add_argument(
        "--rebuild-vectors", action="store_true", help="Rebuild vector embeddings after fetching"
    )
    fetch_parser.add_argument(
        "--vectors-only", action="store_true", help="Only rebuild vectors, skip fetching"
    )

    # Recommend command
    rec_parser = subparsers.add_parser("recommend", help="Get paper recommendations")
    rec_parser.add_argument(
        "-k", "--num-results", type=int, default=10, help="Number of recommendations (default: 10)"
    )
    rec_parser.add_argument(
        "--collection", type=str, help="Base recommendations on a specific Zotero collection"
    )
    rec_parser.add_argument("-o", "--output", type=str, help="Save recommendations to markdown file")
    rec_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Rerank recommendations with a cross-encoder",
    )
    rec_parser.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        help="Disable reranking",
    )
    rec_parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=50,
        help="Number of top candidates to rerank (default: 50)",
    )
    rec_parser.add_argument(
        "--max-age",
        type=positive_int,
        default=None,
        help="Only recommend papers published within this many days",
    )
    rec_parser.set_defaults(rerank=False)

    # Search command
    search_parser = subparsers.add_parser("search", help="Semantic search across papers")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "-k", "--num-results", type=int, default=5, help="Number of results (default: 5)"
    )
    search_parser.add_argument(
        "-s",
        "--source",
        choices=["daily_papers", "zotero"],
        default="daily_papers",
        help="Data source to search",
    )
    search_parser.add_argument("--rag", action="store_true", help="Use RAG to answer the query")
    search_parser.add_argument("--scores", action="store_true", help="Show similarity scores")
    search_parser.add_argument("--collection", type=str, help="Filter by Zotero collection")

    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration info")
    config_parser.add_argument("--data-dir", action="store_true", help="Show data directory path")
    config_parser.add_argument(
        "--check", action="store_true", help="Validate configuration for all operations"
    )

    # Prune command
    prune_parser = subparsers.add_parser("prune", help="Delete old papers from database and vector store")
    prune_parser.add_argument(
        "--older-than",
        type=positive_int,
        required=True,
        help="Delete papers older than this many days",
    )
    prune_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )

    # Digest command
    digest_parser = subparsers.add_parser("digest", help="Send email digest of recommendations")
    digest_parser.add_argument(
        "--days", type=positive_int, default=2,
        help="Number of days to fetch papers (default: 2, to handle timezone differences)"
    )
    digest_parser.add_argument(
        "--arxiv-days",
        type=positive_int,
        default=None,
        help="Number of days to look back for arXiv (default: same as --days). "
             "Useful since arXiv has batch processing delays.",
    )
    digest_parser.add_argument(
        "-k", "--num-results", type=int, default=10, help="Number of recommendations (default: 10)"
    )
    digest_parser.add_argument(
        "--collection", type=str, help="Base recommendations on a specific Zotero collection"
    )
    digest_parser.add_argument(
        "--dry-run", action="store_true", help="Print HTML instead of sending email"
    )
    digest_parser.add_argument(
        "--skip-fetch", action="store_true", help="Skip fetching, use existing papers"
    )
    digest_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking",
    )
    digest_parser.add_argument(
        "--max-age",
        type=positive_int,
        default=None,
        help="Only recommend papers published within this many days (avoids repeats)",
    )
    digest_parser.add_argument(
        "--include-sent-days",
        type=positive_int,
        default=None,
        help="Include papers sent within last N days (to resend a recent digest)",
    )

    args = parser.parse_args()

    # Setup logging based on flags
    from paperfind.logging import setup_logging

    log_level: Optional[str] = args.log_level
    if args.verbose and not log_level:
        log_level = "DEBUG"

    setup_logging(level=log_level, verbose=args.verbose)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Load configuration
    from paperfind.config import load_config
    from paperfind.logging import get_logger

    logger = get_logger(__name__)

    try:
        config_path = load_config(args.config)
        if args.verbose:
            if config_path:
                logger.debug(f"Loaded config from: {config_path}")
            else:
                logger.debug("Using environment variables (no .env file)")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if args.command == "sync":
        from paperfind.fetchers.zotero.sync import run_sync

        run_sync(args.list_collections)

    elif args.command == "embed":
        from paperfind.fetchers.zotero.sync import run_embed

        run_embed(args.collection, args.force)

    elif args.command == "fetch":
        from paperfind.fetchers.fetch_papers import run_fetch

        run_fetch(
            days=args.days,
            arxiv_days=args.arxiv_days,
            sources=args.source,
            rebuild_vectors_flag=args.rebuild_vectors,
            vectors_only=args.vectors_only,
        )

    elif args.command == "recommend":
        from paperfind.search.recommend import run_recommend

        run_recommend(
            num_results=args.num_results,
            collection=args.collection,
            output=args.output,
            rerank=args.rerank,
            rerank_candidates=args.rerank_candidates,
            max_age_days=args.max_age,
        )

    elif args.command == "search":
        from paperfind.search.search import run_search

        run_search(
            query=args.query,
            num_results=args.num_results,
            source=args.source,
            rag=args.rag,
            scores=args.scores,
            collection=args.collection,
        )

    elif args.command == "config":
        from paperfind.config import DATA_DIR, get_config_status, get_loaded_config_path

        if args.data_dir:
            logger.info(DATA_DIR)
        elif args.check:
            status = get_config_status()
            logger.info(f"Data directory: {status['data_dir']}")
            logger.info(f"Config file: {status['config_file'] or 'Not loaded'}")
            logger.info(f"Embedding provider: {status['embedding_provider']}")
            logger.info(f"Embedding model: {status['embedding_model']}")
            logger.info("")
            logger.info("Configuration status:")
            all_valid = True
            for op, missing in status["operations"].items():
                if missing:
                    logger.warning(f"  {op}: Missing {', '.join(missing)}")
                    all_valid = False
                else:
                    logger.info(f"  {op}: OK")
            if all_valid:
                logger.info("")
                logger.info("All configurations valid!")
            else:
                logger.info("")
                logger.info("See .env.example for required variables.")
        else:
            config_file = get_loaded_config_path()
            logger.info(f"Config file: {config_file}")
            logger.info(f"Data directory: {DATA_DIR}")
            logger.info("")
            logger.info("To use a different config: paperfind --config /path/to/.env <command>")
            logger.info("To set data directory: add PAPERFIND_DATA_DIR to your .env file")
            logger.info("Use --check to validate configuration.")

    elif args.command == "digest":
        from paperfind.digest import run_digest

        run_digest(
            days=args.days,
            arxiv_days=args.arxiv_days,
            num_recommendations=args.num_results,
            collection=args.collection,
            dry_run=args.dry_run,
            skip_fetch=args.skip_fetch,
            rerank=args.rerank,
            max_age_days=args.max_age,
            include_sent_days=args.include_sent_days,
        )

    elif args.command == "prune":
        from datetime import date, timedelta

        from paperfind.fetchers.db import get_old_dois, prune_papers
        from paperfind.fetchers.vector import prune_vectors
        from paperfind.logging import get_logger

        logger = get_logger(__name__)
        cutoff_date = date.today() - timedelta(days=args.older_than)

        if args.dry_run:
            dois = get_old_dois(cutoff_date)
            logger.info(f"[Dry run] Would delete {len(dois)} papers older than {cutoff_date}")
            if dois and args.verbose:
                for doi in dois[:10]:
                    logger.info(f"  - {doi}")
                if len(dois) > 10:
                    logger.info(f"  ... and {len(dois) - 10} more")
        else:
            logger.info(f"[Prune] Deleting papers older than {cutoff_date}...")
            deleted_count, deleted_dois = prune_papers(cutoff_date)
            if deleted_dois:
                prune_vectors(deleted_dois)
            logger.info(f"[Prune] Done! Deleted {deleted_count} papers.")


if __name__ == "__main__":
    main()
