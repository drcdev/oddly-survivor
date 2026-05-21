"""Project-wide constants."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SITE_DIR = REPO_ROOT / "_site"
TEMPLATES_DIR = REPO_ROOT / "templates"
STATIC_DIR = REPO_ROOT / "static"
FEATURE_DICT_PATH = DATA_DIR / "feature_dictionary.yml"
ARCHIVE_PATH = DATA_DIR / "archive" / "slugs.txt"

UPSTREAM_OWNER = "doehm"
UPSTREAM_REPO = "survivoR"
UPSTREAM_API = f"https://api.github.com/repos/{UPSTREAM_OWNER}/{UPSTREAM_REPO}/releases/latest"
RAW_BASE = f"https://raw.githubusercontent.com/{UPSTREAM_OWNER}/{UPSTREAM_REPO}"

TABLES = [
    "castaways",
    "castaway_details",
    "season_summary",
    "episodes",
    "vote_history",
    "jury_votes",
    "challenge_results",
    "challenge_description",
    "confessionals",
    "advantage_movement",
    "boot_mapping",
    "tribe_mapping",
    "tribe_colours",
    "season_palettes",
]

MAX_US_SEASON = 50
MIN_OVERLAP_SEASONS = 30
TOP_N = 75
ALL_N = 300

PEARSON_THRESHOLD = 0.45
P_VALUE_THRESHOLD = 0.05
