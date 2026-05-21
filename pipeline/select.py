"""Score and rank correlations by 'spuriousness'."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.config import ALL_N, P_VALUE_THRESHOLD, PEARSON_THRESHOLD, TOP_N
from pipeline.features import REGISTRY


def _meta_map() -> dict[str, dict]:
    return {m.key: {"domain": m.domain, "source_tables": set(m.source_tables)} for m in REGISTRY}


def _monotonicity_penalty(features: pd.DataFrame, a: str, b: str) -> float:
    """Penalise pairs where both series are nearly-monotone in season number.

    Those are just 'things drifted over time' — boring, not spurious-looking.
    """
    seasons = features.index.to_numpy(dtype=float)

    def monotone_score(col: str) -> float:
        x = features[col]
        mask = x.notna()
        if mask.sum() < 5:
            return 0.0
        r = abs(float(np.corrcoef(seasons[mask], x[mask].to_numpy(dtype=float))[0, 1]))
        return r

    ma = monotone_score(a)
    mb = monotone_score(b)
    if ma > 0.9 and mb > 0.9:
        return 0.4
    if ma > 0.8 and mb > 0.8:
        return 0.7
    return 1.0


def score_pairs(pairs: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    meta = _meta_map()
    rows = []
    for _, row in pairs.iterrows():
        a, b = row["a"], row["b"]
        ma, mb = meta[a], meta[b]
        # Skip pairs from the same source table — likely trivially related.
        if ma["source_tables"] & mb["source_tables"]:
            shared_only = ma["source_tables"] == mb["source_tables"]
            if shared_only:
                continue
        same_domain = ma["domain"] == mb["domain"]
        domain_distance = 0.3 if same_domain else 1.0
        sign_agreement = 1.0 if np.sign(row["r_pearson"]) == np.sign(row["r_spearman"]) else 0.5
        mono_pen = _monotonicity_penalty(features, a, b)
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
