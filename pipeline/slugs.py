"""Stable slug generation for correlation detail pages."""

from __future__ import annotations

import re


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text


def pair_slug(a: str, b: str) -> str:
    """Order-independent slug for a feature pair."""
    lo, hi = sorted([a, b])
    return f"{slugify(lo)}--vs--{slugify(hi)}"
