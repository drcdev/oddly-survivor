from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.correlate import all_pairs
from pipeline.features import build
from pipeline.select import score_pairs


def test_pairs_have_consistent_columns(probe_tag, chdir_repo_root):
    features = build(probe_tag)
    pairs = all_pairs(features)
    assert {"a", "b", "n", "r_pearson", "p_pearson", "r_spearman", "p_spearman"} <= set(pairs.columns)
    assert (pairs["a"] != pairs["b"]).all()


def test_score_pairs_filters_and_sorts(probe_tag, chdir_repo_root):
    features = build(probe_tag)
    pairs = all_pairs(features)
    scored = score_pairs(pairs, features)
    assert scored["score"].is_monotonic_decreasing
    assert (scored["p_pearson"] <= 0.05).all()


def test_known_correlation_orders_top_first():
    # Two crafted series with near-perfect correlation outrank uncorrelated noise.
    rng = np.random.default_rng(0)
    n = 50
    x = np.arange(n, dtype=float)
    df = pd.DataFrame(
        {
            "alpha": x + rng.normal(0, 0.5, n),
            "beta": x + rng.normal(0, 0.5, n),
            "noise1": rng.normal(0, 1, n),
            "noise2": rng.normal(0, 1, n),
        },
        index=range(1, n + 1),
    )
    pairs = all_pairs(df)
    pairs_sorted = pairs.reindex(pairs["r_pearson"].abs().sort_values(ascending=False).index)
    top = pairs_sorted.iloc[0]
    assert {top["a"], top["b"]} == {"alpha", "beta"}
