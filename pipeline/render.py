"""Render the static site from the scored correlations."""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from pipeline.charts import ChartSpec, render_png, render_svg, render_thumbnail_svg
from pipeline.config import (
    ARCHIVE_PATH,
    SITE_DIR,
    STATIC_DIR,
    TEMPLATES_DIR,
    TOP_N,
)
from pipeline.features import REGISTRY
from pipeline.palette import colors_for_pair
from pipeline.slugs import pair_slug

log = logging.getLogger(__name__)


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "html.j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _meta_lookup() -> dict[str, dict]:
    return {
        m.key: {
            "label": m.label,
            "description": m.description,
            "domain": m.domain,
            "source_tables": list(m.source_tables),
            "units": m.units,
        }
        for m in REGISTRY
    }


def _load_archive() -> set[str]:
    if not ARCHIVE_PATH.exists():
        return set()
    return {line.strip() for line in ARCHIVE_PATH.read_text().splitlines() if line.strip()}


def _save_archive(slugs: set[str]) -> None:
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE_PATH.write_text("\n".join(sorted(slugs)) + "\n")


def _write_csv(path: Path, seasons, a_vals, b_vals, label_a: str, label_b: str) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["season", label_a, label_b])
        for s, a, b in zip(seasons, a_vals, b_vals, strict=False):
            w.writerow(
                [
                    int(s),
                    "" if pd.isna(a) else a,
                    "" if pd.isna(b) else b,
                ]
            )


def _build_card_record(idx: int, row: pd.Series, features: pd.DataFrame, meta: dict, tag: str) -> dict:
    a, b = row["a"], row["b"]
    color_a, color_b = colors_for_pair(a, b, tag)
    return {
        "rank": idx + 1,
        "a": a,
        "b": b,
        "slug": pair_slug(a, b),
        "label_a": meta[a]["label"],
        "label_b": meta[b]["label"],
        "color_a": color_a,
        "color_b": color_b,
        "r_pearson": row["r_pearson"],
        "p_pearson": row["p_pearson"],
        "r_spearman": row["r_spearman"],
        "p_spearman": row["p_spearman"],
        "n": int(row["n"]),
        "score": row["score"],
    }


def render(
    features: pd.DataFrame,
    scored: pd.DataFrame,
    archive_pool: pd.DataFrame,
    tag: str,
    site_dir: Path | None = None,
    pages_url: str = "",
) -> None:
    site = site_dir or SITE_DIR
    if site.exists():
        shutil.rmtree(site)
    site.mkdir(parents=True)
    shutil.copytree(STATIC_DIR, site, dirs_exist_ok=True)

    env = _env()
    meta = _meta_lookup()

    n_features = len(features.columns)
    n_seasons = int(features.shape[0])
    n_pairs_tested = (n_features * (n_features - 1)) // 2

    archived_slugs = _load_archive()

    featured = scored.head(TOP_N).reset_index(drop=True)
    featured_records = [_build_card_record(i, row, features, meta, tag) for i, row in featured.iterrows()]
    all_records = [_build_card_record(i, row, features, meta, tag) for i, row in archive_pool.iterrows()]
    all_slugs_this_build = {rec["slug"] for rec in all_records}
    archived_slugs |= all_slugs_this_build
    _save_archive(archived_slugs)

    # index
    cards_for_index = []
    for rec in featured_records:
        a, b = rec["a"], rec["b"]
        thumb_svg = render_thumbnail_svg(
            features.index,
            features[a],
            features[b],
            ChartSpec(rec["label_a"], rec["label_b"], rec["color_a"], rec["color_b"], meta[a]["units"], meta[b]["units"]),
        )
        cards_for_index.append({**rec, "thumb_svg": thumb_svg})

    tpl_ctx = {
        "root": "/",
        "build_tag": tag,
        "n_features": n_features,
        "n_seasons": n_seasons,
    }

    (site / "index.html").write_text(
        env.get_template("index.html.j2").render(
            cards=cards_for_index,
            **tpl_ctx,
        )
    )

    # detail pages for the full archive_pool (so dropped-but-archived pairs still resolve)
    (site / "c").mkdir()
    n_pairs_for_bonferroni = max(n_pairs_tested, 1)
    ordered = all_records

    for idx, rec in enumerate(ordered):
        a, b = rec["a"], rec["b"]
        spec = ChartSpec(
            rec["label_a"],
            rec["label_b"],
            rec["color_a"],
            rec["color_b"],
            meta[a]["units"],
            meta[b]["units"],
        )
        chart_svg = render_svg(features.index, features[a], features[b], spec)
        page_dir = site / "c" / rec["slug"]
        page_dir.mkdir()

        render_png(features.index, features[a], features[b], spec, str(page_dir / "og.png"))

        rows = [
            {
                "season": int(s),
                "a": "—" if pd.isna(av) else _fmt(av),
                "b": "—" if pd.isna(bv) else _fmt(bv),
            }
            for s, av, bv in zip(features.index, features[a].tolist(), features[b].tolist(), strict=False)
        ]

        _write_csv(page_dir / "data.csv", features.index, features[a].tolist(), features[b].tolist(), rec["label_a"], rec["label_b"])

        prev_rec = ordered[idx - 1] if idx > 0 else None
        next_rec = ordered[idx + 1] if idx + 1 < len(ordered) else None

        page = env.get_template("correlation.html.j2").render(
            root="/",
            build_tag=tag,
            label_a=rec["label_a"],
            label_b=rec["label_b"],
            color_a=rec["color_a"],
            color_b=rec["color_b"],
            desc_a=meta[a]["description"],
            desc_b=meta[b]["description"],
            domain_a=meta[a]["domain"],
            domain_b=meta[b]["domain"],
            sources_a=", ".join(meta[a]["source_tables"]),
            sources_b=", ".join(meta[b]["source_tables"]),
            n=rec["n"],
            r_pearson=rec["r_pearson"],
            p_pearson=rec["p_pearson"],
            r_spearman=rec["r_spearman"],
            p_spearman=rec["p_spearman"],
            p_bonferroni=min(rec["p_pearson"] * n_pairs_for_bonferroni, 1.0),
            rank=rec["rank"],
            chart_svg=chart_svg,
            rows=rows,
            prev=prev_rec,
            next=next_rec,
            canonical_url=f"{pages_url}/c/{rec['slug']}/" if pages_url else "",
        )
        (page_dir / "index.html").write_text(page)

    # all page
    (site / "all").mkdir()
    (site / "all" / "index.html").write_text(
        env.get_template("all.html.j2").render(
            rows=all_records,
            **tpl_ctx,
        )
    )

    # about page
    (site / "about").mkdir()
    (site / "about" / "index.html").write_text(
        env.get_template("about.html.j2").render(
            n_pairs=n_pairs_tested,
            top_n=TOP_N,
            **tpl_ctx,
        )
    )

    (site / ".last_version").write_text(tag.strip() + "\n")

    log.info("rendered %d featured + %d total correlation pages", len(featured_records), len(all_records))


def _fmt(v: float) -> str:
    if abs(v) >= 100 or v == int(v):
        return f"{v:,.0f}"
    if abs(v) >= 1:
        return f"{v:,.2f}"
    return f"{v:.3f}"
