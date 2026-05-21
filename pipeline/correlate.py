"""Compute all pairwise correlations among season features."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.config import MIN_OVERLAP_SEASONS


def all_pairs(features: pd.DataFrame) -> pd.DataFrame:
    cols = list(features.columns)
    rows = []
    for a, b in combinations(cols, 2):
        x = features[a]
        y = features[b]
        mask = x.notna() & y.notna()
        n = int(mask.sum())
        if n < MIN_OVERLAP_SEASONS:
            continue
        xv = x[mask].to_numpy(dtype=float)
        yv = y[mask].to_numpy(dtype=float)
        if np.std(xv) == 0 or np.std(yv) == 0:
            continue
        p_res = stats.pearsonr(xv, yv)
        s_res = stats.spearmanr(xv, yv)
        rows.append(
            {
                "a": a,
                "b": b,
                "n": n,
                "r_pearson": float(p_res.statistic),
                "p_pearson": float(p_res.pvalue),
                "r_spearman": float(s_res.statistic),
                "p_spearman": float(s_res.pvalue),
            }
        )
    return pd.DataFrame(rows)
