"""Pull `.rda` files from upstream survivoR releases and load them as DataFrames."""

from __future__ import annotations

import logging
from functools import cache

import pandas as pd
import pyreadr
import requests

from pipeline.config import (
    CACHE_DIR,
    MAX_US_SEASON,
    RAW_BASE,
    TABLES,
    UPSTREAM_API,
)

log = logging.getLogger(__name__)


class SchemaDriftError(RuntimeError):
    """Raised when an upstream table is missing an expected column."""


def latest_upstream_tag() -> str:
    resp = requests.get(UPSTREAM_API, timeout=30)
    resp.raise_for_status()
    return resp.json()["tag_name"]


def resolve_tag(tag: str) -> str:
    """`latest` is resolved to the actual tag; anything else passes through."""
    if tag in ("latest", "", None):
        return latest_upstream_tag()
    return tag


def fetch_table(name: str, tag: str) -> str:
    """Download `<name>.rda` if not cached. Returns local path."""
    if name not in TABLES:
        raise ValueError(f"unknown table {name!r}; not in TABLES")
    tag_dir = CACHE_DIR / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    dest = tag_dir / f"{name}.rda"
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)
    url = f"{RAW_BASE}/{tag}/data/{name}.rda"
    log.info("downloading %s", url)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return str(dest)


def _read_rda(path: str) -> pd.DataFrame:
    try:
        result = pyreadr.read_r(path)
    except pyreadr.custom_errors.LibrdataError:
        import rdata

        parsed = rdata.parser.parse_file(path)
        converted = rdata.conversion.convert(parsed)
        result = {k: converted[k] for k in converted}
    if not result:
        raise RuntimeError(f"empty .rda at {path}")
    key = next(iter(result))
    return result[key]


@cache
def load_table(name: str, tag: str) -> pd.DataFrame:
    path = fetch_table(name, tag)
    df = _read_rda(path)
    for col in df.columns:
        if hasattr(df[col], "cat"):
            df[col] = df[col].astype(str)
    return df


EXPECTED_COLUMNS: dict[str, list[str]] = {
    "castaways": ["version", "season", "castaway_id", "age", "winner", "place", "city", "state"],
    "castaway_details": ["castaway_id", "gender", "bipoc", "lgbt"],
    "season_summary": [
        "version",
        "season",
        "season_name",
        "n_cast",
        "n_tribes",
        "n_jury",
        "viewers_premiere",
        "viewers_finale",
        "viewers_mean",
        "premiered",
        "ended",
    ],
    "episodes": [
        "version",
        "season",
        "episode",
        "episode_date",
        "viewers",
        "imdb_rating",
        "episode_length",
    ],
    "vote_history": ["version", "season", "episode", "castaway_id", "vote", "voted_out_id", "tie"],
    "jury_votes": ["version", "season", "castaway_id", "finalist_id", "vote"],
    "challenge_results": [
        "version",
        "season",
        "castaway_id",
        "challenge_type",
        "won_individual_immunity",
        "won_individual_reward",
    ],
    "challenge_description": [
        "version",
        "season",
        "challenge_id",
        "endurance",
        "puzzle",
        "strength",
        "water",
        "fire",
    ],
    "confessionals": ["version", "season", "castaway_id", "confessional_count"],
    "advantage_movement": ["version", "season", "castaway_id", "event"],
    "boot_mapping": ["version", "season", "episode", "castaway_id"],
    "tribe_mapping": ["version", "season", "castaway_id", "tribe"],
    "tribe_colours": ["version", "season", "tribe", "tribe_colour"],
    "season_palettes": ["version", "season", "palette"],
}


def _validate(name: str, df: pd.DataFrame) -> None:
    expected = EXPECTED_COLUMNS.get(name, [])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SchemaDriftError(f"table {name!r} missing expected columns: {missing}")


def load_us(name: str, tag: str) -> pd.DataFrame:
    """Load a table filtered to US Survivor, seasons <= MAX_US_SEASON."""
    df = load_table(name, tag).copy()
    _validate(name, df)
    if "version" in df.columns:
        df = df[df["version"] == "US"]
    if "season" in df.columns:
        df["season"] = pd.to_numeric(df["season"], errors="coerce")
        df = df[df["season"] <= MAX_US_SEASON]
        df["season"] = df["season"].astype("Int64")
    return df.reset_index(drop=True)


def load_all(tag: str) -> dict[str, pd.DataFrame]:
    return {name: load_us(name, tag) for name in TABLES}
