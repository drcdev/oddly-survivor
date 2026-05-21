from __future__ import annotations

from pipeline.features import REGISTRY, build


def test_build_yields_50_rows(probe_tag, chdir_repo_root):
    df = build(probe_tag)
    assert df.shape[0] == 50


def test_every_column_has_registry_entry(probe_tag, chdir_repo_root):
    df = build(probe_tag)
    keys = {m.key for m in REGISTRY}
    assert set(df.columns) <= keys


def test_no_duplicate_keys():
    keys = [m.key for m in REGISTRY]
    assert len(keys) == len(set(keys))


def test_registry_has_required_fields():
    for m in REGISTRY:
        assert m.label
        assert m.description
        assert m.domain
        assert len(m.source_tables) >= 1


def test_no_unexpected_all_nan_columns(probe_tag, chdir_repo_root):
    df = build(probe_tag)
    all_nan = [c for c in df.columns if df[c].isna().all()]
    assert all_nan == [], f"all-NaN columns: {all_nan}"
