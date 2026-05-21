"""Score and rank correlations by 'spuriousness'."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.config import ALL_N, P_VALUE_THRESHOLD, PEARSON_THRESHOLD, TOP_N
from pipeline.features import REGISTRY


def _meta_map() -> dict[str, dict]:
    return {m.key: {"domain": m.domain, "source_tables": set(m.source_tables)} for m in REGISTRY}


def _season_monotonicity(features: pd.DataFrame) -> dict[str, float]:
    """For each feature, |Pearson correlation with season number|."""
    seasons = features.index.to_numpy(dtype=float)
    out: dict[str, float] = {}
    for col in features.columns:
        x = features[col]
        mask = x.notna()
        if mask.sum() < 5:
            out[col] = 0.0
            continue
        std = x[mask].std()
        if std == 0 or np.isnan(std):
            out[col] = 0.0
            continue
        out[col] = abs(float(np.corrcoef(seasons[mask], x[mask].to_numpy(dtype=float))[0, 1]))
    return out


def _monotonicity_penalty(mono_a: float, mono_b: float) -> float:
    """Multiplier ≤ 1 that crashes when both variables drift with time.

    Pairs of strongly time-trending features just reflect 'Survivor changed
    over the decades' — boring, not spurious-looking. We want at least one
    variable to be independent of time.
    """
    return (1.0 - min(mono_a, mono_b)) ** 2


def score_pairs(pairs: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    meta = _meta_map()
    monotonicity = _season_monotonicity(features)
    rows = []
    for _, row in pairs.iterrows():
        a, b = row["a"], row["b"]
        ma, mb = meta[a], meta[b]
        if ma["source_tables"] & mb["source_tables"]:
            shared_only = ma["source_tables"] == mb["source_tables"]
            if shared_only:
                continue
        same_domain = ma["domain"] == mb["domain"]
        domain_distance = 0.3 if same_domain else 1.0
        sign_agreement = 1.0 if np.sign(row["r_pearson"]) == np.sign(row["r_spearman"]) else 0.5
        mono_pen = _monotonicity_penalty(monotonicity[a], monotonicity[b])
        score = (
            abs(row["r_pearson"])
            * sign_agreement
            * ((1.0 - row["p_pearson"]) ** 0.5)
            * domain_distance
            * mono_pen
        )
        rows.append(
            {
                **row.to_dict(),
                "domain_a": ma["domain"],
                "domain_b": mb["domain"],
                "score": float(score),
                "same_domain": same_domain,
                "monotonicity_penalty": mono_pen,
                "monotonicity_a": monotonicity[a],
                "monotonicity_b": monotonicity[b],
            }
        )
    df = pd.DataFrame(rows)
    df = df[df["p_pearson"] <= P_VALUE_THRESHOLD]
    df = df[df["r_pearson"].abs() >= PEARSON_THRESHOLD]
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def featured(scored: pd.DataFrame) -> pd.DataFrame:
    return scored.head(TOP_N).copy()


def archive_pool(scored: pd.DataFrame) -> pd.DataFrame:
    return scored.head(ALL_N).copy()
