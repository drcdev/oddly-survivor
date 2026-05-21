from __future__ import annotations

from pathlib import Path

from pipeline.correlate import all_pairs
from pipeline.features import build
from pipeline.render import render
from pipeline.select import archive_pool, score_pairs


def test_render_builds_site(probe_tag, chdir_repo_root, tmp_path):
    features = build(probe_tag)
    pairs = all_pairs(features)
    scored = score_pairs(pairs, features)
    pool = archive_pool(scored)
    site = tmp_path / "site"
    render(features, scored, pool, tag=probe_tag, site_dir=site)

    assert (site / "index.html").exists()
    assert (site / "all" / "index.html").exists()
    assert (site / "about" / "index.html").exists()
    assert (site / ".last_version").read_text().strip() == probe_tag
    assert (site / "style.css").exists()

    detail_pages = list((site / "c").glob("*/index.html"))
    assert len(detail_pages) == len(pool)
    for page in detail_pages[:5]:
        csv_file = page.parent / "data.csv"
        og_png = page.parent / "og.png"
        assert csv_file.exists()
        assert og_png.exists()
        assert og_png.stat().st_size > 1024


def test_internal_links_resolve(probe_tag, chdir_repo_root, tmp_path):
    import re

    features = build(probe_tag)
    scored = score_pairs(all_pairs(features), features)
    pool = archive_pool(scored)
    site = tmp_path / "site"
    render(features, scored, pool, tag=probe_tag, site_dir=site)

    index_html = (site / "index.html").read_text()
    hrefs = re.findall(r'href="(/c/[^"]+)/"', index_html)
    for href in hrefs:
        slug_dir = site / Path(href.lstrip("/"))
        assert (slug_dir / "index.html").exists(), f"missing {href}"
