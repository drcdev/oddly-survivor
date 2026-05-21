from __future__ import annotations

import pandas as pd
import pytest

from pipeline import ingest
from pipeline.config import MAX_US_SEASON


def test_load_us_filters_version_and_season(probe_tag, chdir_repo_root):
    cast = ingest.load_us("castaways", probe_tag)
    assert (cast["version"] == "US").all()
    assert cast["season"].max() <= MAX_US_SEASON
    assert cast["season"].dtype == "Int64"


def test_load_us_season_summary_50_rows(probe_tag, chdir_repo_root):
    ss = ingest.load_us("season_summary", probe_tag)
    assert ss["season"].nunique() == 50


def test_schema_drift_error(monkeypatch, probe_tag, chdir_repo_root):
    real_load = ingest.load_table
    monkeypatch.setattr(
        ingest, "load_table", lambda name, tag: real_load(name, tag).drop(columns=["age"]) if name == "castaways" else real_load(name, tag)
    )
    with pytest.raises(ingest.SchemaDriftError):
        ingest.load_us("castaways", probe_tag)


def test_resolve_tag_passes_through_non_latest():
    assert ingest.resolve_tag("v2.3.11") == "v2.3.11"


def test_load_all_returns_all_tables(probe_tag, chdir_repo_root):
    tables = ingest.load_all(probe_tag)
    assert isinstance(tables["castaways"], pd.DataFrame)
    assert len(tables) >= 10
