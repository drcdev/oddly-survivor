"""Pick chart colors from survivoR season palettes."""

from __future__ import annotations

import hashlib
from functools import lru_cache

from pipeline.ingest import load_us

FALLBACK_PALETTE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
]


def _luminance(hex_color: str) -> float:
    r, g, b = _to_rgb(hex_color)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


@lru_cache(maxsize=1)
def season_palettes(tag: str) -> dict[int, list[str]]:
    df = load_us("season_palettes", tag)
    out: dict[int, list[str]] = {}
    for season, group in df.groupby("season"):
        colors: list[str] = []
        for val in group["palette"].tolist():
            if isinstance(val, list):
                colors.extend(str(c) for c in val)
            elif isinstance(val, str):
                if val.startswith("#"):
                    colors.append(val)
                else:
                    parts = [p.strip() for p in val.replace("[", "").replace("]", "").split(",")]
                    colors.extend(p for p in parts if p.startswith("#"))
        colors = [c for c in colors if isinstance(c, str) and c.startswith("#") and len(c) in (4, 7)]
        # Drop near-black and near-white colors that look bad as line/text colors.
        colors = [c for c in colors if 0.12 < _luminance(c) < 0.78]
        out[int(season)] = colors or FALLBACK_PALETTE
    return out


def _hash_to_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def colors_for_pair(feature_a: str, feature_b: str, tag: str) -> tuple[str, str]:
    """Deterministic pair of distinct hex colors drawn from season palettes."""
    palettes = season_palettes(tag)
    seasons = sorted(palettes.keys())
    if not seasons:
        return FALLBACK_PALETTE[0], FALLBACK_PALETTE[1]
    sa = seasons[_hash_to_int(feature_a) % len(seasons)]
    sb = seasons[_hash_to_int(feature_b) % len(seasons)]
    ca_palette = palettes.get(sa) or FALLBACK_PALETTE
    cb_palette = palettes.get(sb) or FALLBACK_PALETTE
    ca = ca_palette[_hash_to_int(feature_a + "x") % len(ca_palette)]
    cb = cb_palette[_hash_to_int(feature_b + "y") % len(cb_palette)]
    if ca.lower() == cb.lower():
        cb = cb_palette[(_hash_to_int(feature_b + "y") + 1) % len(cb_palette)]
    return ca, cb


def _to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def is_dark(hex_color: str) -> bool:
    r, g, b = _to_rgb(hex_color)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return luminance < 0.5
