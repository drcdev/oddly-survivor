"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_CACHE = REPO_ROOT / "data" / "cache" / "probe"


@pytest.fixture(scope="session")
def probe_tag() -> str:
    """A tag whose .rda files we expect to already be cached at data/cache/probe.

    Tests skip themselves when the cache is missing, so the suite still passes
    in a fresh checkout that hasn't downloaded data yet.
    """
    if not PROBE_CACHE.exists() or not any(PROBE_CACHE.glob("*.rda")):
        pytest.skip("probe cache missing; run `uv run python -m pipeline.cli fetch --tag probe` once")
    return "probe"


@pytest.fixture
def chdir_repo_root(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    yield REPO_ROOT
