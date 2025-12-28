"""
Command-line interface for Paperfind.

Usage:
    paperfind sync                     # Sync Zotero library
    paperfind fetch                    # Fetch papers from all sources
    paperfind recommend                # Get paper recommendations
    paperfind search "query"           # Semantic search
    paperfind digest                   # Send email digest of recommendations
"""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="paperfind",
        description="Paper recommendation system based on your Zotero library",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync Zotero library to local database")
    sync_parser.add_argument("--collection", type=str, help="Sync a specific collection by name")
    sync_parser.add_argument(
        "--list-collections", action="store_true", help="List all collections in your library"
    )

    # Fetch command
    from paperfind.config import BIORXIV_CATEGORIES

    fetch_parser = subparsers.add_parser("fetch", help="Fetch papers from external sources")
    fetch_parser.add_argument(
        "--days", type=int, default=1, help="Number of days to look back (default: 1)"
    )
    fetch_parser.add_argument(
        "--source",
        action="append",
        choices=["crossref", "biorxiv", "medrxiv", "arxiv"],
        help="Source(s) to fetch from (default: all)",
    )
    fetch_parser.add_argument(
        "--biorxiv-category",
        choices=BIORXIV_CATEGORIES,
        help="bioRxiv category filter (default: all)",
    )
    fetch_parser.add_argument(
        "--medrxiv-category",
        choices=BIORXIV_CATEGORIES,
        help="medRxiv category filter (default: all)",
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
    search_parser.add_argument("--project-id", type=int, help="Filter by Zotero project ID")

    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration info")
    config_parser.add_argument("--data-dir", action="store_true", help="Show data directory path")

    # Digest command
    digest_parser = subparsers.add_parser("digest", help="Send email digest of recommendations")
    digest_parser.add_argument(
        "--days", type=int, default=1, help="Number of days to fetch papers (default: 1)"
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

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "sync":
        from paperfind.fetchers.zotero.sync import run_sync

        run_sync(args.collection, args.list_collections)

    elif args.command == "fetch":
        from paperfind.fetchers.fetch_papers import run_fetch

        run_fetch(
            days=args.days,
            sources=args.source,
            biorxiv_category=args.biorxiv_category,
            medrxiv_category=args.medrxiv_category,
            rebuild_vectors_flag=args.rebuild_vectors,
            vectors_only=args.vectors_only,
        )

    elif args.command == "recommend":
        from paperfind.search.recommend import run_recommend

        run_recommend(
            num_results=args.num_results,
            collection=args.collection,
            output=args.output,
        )

    elif args.command == "search":
        from paperfind.search.search import run_search

        run_search(
            query=args.query,
            num_results=args.num_results,
            source=args.source,
            rag=args.rag,
            scores=args.scores,
            project_id=args.project_id,
        )

    elif args.command == "config":
        from paperfind.config import DATA_DIR

        if args.data_dir:
            print(DATA_DIR)
        else:
            print(f"Data directory: {DATA_DIR}")
            print(f"\nTo use a different location, set PAPERFIND_DATA_DIR environment variable.")
            print(f"\nPlace your .env file in the data directory or current working directory.")

    elif args.command == "digest":
        from paperfind.digest import run_digest

        run_digest(
            days=args.days,
            num_recommendations=args.num_results,
            collection=args.collection,
            dry_run=args.dry_run,
            skip_fetch=args.skip_fetch,
        )


if __name__ == "__main__":
    main()
