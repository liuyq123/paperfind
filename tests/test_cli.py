"""Tests for CLI commands and module imports.

These tests ensure that:
1. All modules can be imported without errors
2. CLI commands can load their dependencies
3. CLI help commands work correctly
"""

import shutil
import subprocess

import pytest


class TestModuleImports:
    """Test that all modules can be imported without errors."""

    def test_zotero_module_imports(self):
        """Ensure Zotero module exports are importable."""
        from paperfind.fetchers.zotero import (
            run_sync,
            sync_project,
            list_collections,
            rebuild_all_vectors,
            init_db,
            get_conn,
            get_or_create_project,
            fetch_collections,
            fetch_items_for_project,
            get_vectordb,
            rebuild_vectors_for_project,
        )

    def test_digest_module_imports(self):
        """Ensure digest module exports are importable."""
        from paperfind.digest import run_digest
        from paperfind.digest.email import send_email
        from paperfind.digest.template import render_digest

    def test_fetchers_module_imports(self):
        """Ensure fetcher modules are importable."""
        from paperfind.fetchers.fetch_papers import fetch_all, run_fetch
        from paperfind.fetchers.vector import rebuild_vectors, upsert_vectors_for_dois

    def test_search_module_imports(self):
        """Ensure search modules are importable."""
        from paperfind.search.recommend import get_recommendations, run_recommend
        from paperfind.search.search import search, run_search

    def test_cli_main_importable(self):
        """Ensure CLI main function is importable."""
        from paperfind.cli import main


@pytest.mark.skipif(
    shutil.which("paperfind") is None,
    reason="paperfind CLI not installed",
)
class TestCLICommands:
    """Test that CLI commands work correctly."""

    def test_cli_help(self):
        """Test that main help command works."""
        result = subprocess.run(
            ["paperfind", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "paperfind" in result.stdout
        assert "sync" in result.stdout
        assert "fetch" in result.stdout
        assert "recommend" in result.stdout

    def test_sync_help(self):
        """Test that sync --help works."""
        result = subprocess.run(
            ["paperfind", "sync", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--collection" in result.stdout
        assert "--list-collections" in result.stdout

    def test_fetch_help(self):
        """Test that fetch --help works."""
        result = subprocess.run(
            ["paperfind", "fetch", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--days" in result.stdout
        assert "--rebuild-vectors" in result.stdout

    def test_recommend_help(self):
        """Test that recommend --help works."""
        result = subprocess.run(
            ["paperfind", "recommend", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--num-results" in result.stdout
        assert "--rerank" in result.stdout

    def test_search_help(self):
        """Test that search --help works."""
        result = subprocess.run(
            ["paperfind", "search", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "query" in result.stdout
        assert "--rag" in result.stdout

    def test_digest_help(self):
        """Test that digest --help works."""
        result = subprocess.run(
            ["paperfind", "digest", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--dry-run" in result.stdout
        assert "--skip-fetch" in result.stdout

    def test_config_help(self):
        """Test that config --help works."""
        result = subprocess.run(
            ["paperfind", "config", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--data-dir" in result.stdout
