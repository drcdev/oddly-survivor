"""Command-line entry point for the pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
import time

from pipeline.config import SITE_DIR, TABLES
from pipeline.correlate import all_pairs
from pipeline.features import build as build_features
from pipeline.ingest import fetch_table, resolve_tag
from pipeline.render import render
from pipeline.select import archive_pool, score_pairs

log = logging.getLogger("pipeline")


def cmd_fetch(args) -> int:
    tag = resolve_tag(args.tag)
    log.info("fetching tag %s", tag)
    for name in TABLES:
        fetch_table(name, tag)
    print(tag)
    return 0


def cmd_features(args) -> int:
    tag = resolve_tag(args.tag)
    df = build_features(tag)
    df.to_parquet(f"data/season_features_{tag}.parquet")
    print(f"wrote {df.shape[0]} seasons x {df.shape[1]} features")
    return 0


def cmd_correlate(args) -> int:
    tag = resolve_tag(args.tag)
    features = build_features(tag)
    pairs = all_pairs(features)
    scored = score_pairs(pairs, features)
    pool = archive_pool(scored)
    pool.to_parquet(f"data/correlations_{tag}.parquet")
    print(f"scored {len(scored)} pairs (top {len(pool)} retained)")
    return 0


def cmd_build(args) -> int:
    t0 = time.time()
    tag = resolve_tag(args.tag)
    log.info("building site for tag %s", tag)
    for name in TABLES:
        fetch_table(name, tag)
    features = build_features(tag)
    pairs = all_pairs(features)
    scored = score_pairs(pairs, features)
    pool = archive_pool(scored)
    render(features, scored, pool, tag=tag, site_dir=SITE_DIR, pages_url=args.pages_url)
    log.info("done in %.1fs", time.time() - t0)
    print(f"site built at {SITE_DIR} for {tag}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(prog="pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch")
    p_fetch.add_argument("--tag", default="latest")
    p_fetch.set_defaults(func=cmd_fetch)

    p_feat = sub.add_parser("features")
    p_feat.add_argument("--tag", default="latest")
    p_feat.set_defaults(func=cmd_features)

    p_corr = sub.add_parser("correlate")
    p_corr.add_argument("--tag", default="latest")
    p_corr.set_defaults(func=cmd_correlate)

    p_build = sub.add_parser("build")
    p_build.add_argument("--tag", default="latest")
    p_build.add_argument("--pages-url", default="", help="absolute base URL of the deployed Pages site, used in OG tags")
    p_build.set_defaults(func=cmd_build)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
