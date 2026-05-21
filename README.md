# oddly-survivor

Spurious correlations in 50 seasons of US Survivor data, in the style of
[tylervigen.com/spurious-correlations](https://www.tylervigen.com/spurious-correlations).

Data comes from the [`doehm/survivoR`](https://github.com/doehm/survivoR) R
package (MIT licensed). A weekly GitHub Action polls upstream releases and
rebuilds the site to [GitHub Pages](https://drcdev.github.io/oddly-survivor)
whenever a new `survivoR` tag ships.

## How it works

1. **Ingest** — `pyreadr` reads `.rda` files directly from
   `raw.githubusercontent.com/doehm/survivoR/<tag>/data/`, pinned to a release
   tag. Filtered to `version == "US"` and `season <= 50`.
2. **Feature engineering** — `pipeline/features.py` aggregates the raw
   per-castaway / per-episode / per-vote tables into one row per season with
   ~50 numeric columns, each tagged with a "domain" (demographic, gameplay,
   viewership, name/text, …).
3. **Correlate** — every pair of columns gets a Pearson and Spearman
   correlation with p-values.
4. **Score** — a "spuriousness" score favors high `|r|`, low p-value,
   conceptually distant variable domains, and penalizes pairs that are both
   nearly-monotone in season number (those are just "things that drifted over
   time" — boring).
5. **Render** — Jinja2 + inline matplotlib SVG charts produce a static site.
   Top 75 land on the index; ~100 are kept as browseable detail pages.

## Local development

```sh
uv sync
uv run python -m pipeline.cli build --tag v2.3.9
uv run python -m http.server -d _site 8000
# open http://localhost:8000
```

CLI subcommands:

| command | purpose |
|---|---|
| `fetch --tag <t>` | download `.rda` files into `data/cache/<t>/` |
| `features --tag <t>` | write `data/season_features_<t>.parquet` |
| `correlate --tag <t>` | write `data/correlations_<t>.parquet` |
| `build --tag <t>` | run the full pipeline and render `_site/` |

`--tag latest` resolves via the GitHub API; otherwise pass an exact release
tag like `v2.3.9`.

## Tests

```sh
uv run pytest
uv run ruff check pipeline tests
```

Tests require a one-time data fetch:

```sh
mkdir -p data/cache/probe
for t in castaways castaway_details season_summary episodes vote_history \
         jury_votes challenge_results challenge_description confessionals \
         advantage_movement boot_mapping tribe_mapping tribe_colours \
         season_palettes; do
  curl -fsSL "https://raw.githubusercontent.com/doehm/survivoR/master/data/$t.rda" \
    -o "data/cache/probe/$t.rda"
done
```

## Auto-update

- **`.github/workflows/check-upstream.yml`** runs weekly. It compares the
  latest `survivoR` release tag against the `/.last_version` file on the live
  Pages site and dispatches the builder if they differ.
- **`.github/workflows/build-and-deploy.yml`** runs the pipeline and publishes
  to GitHub Pages via `actions/deploy-pages` (no `gh-pages` branch).
- **`.github/workflows/ci.yml`** lints and tests every PR.

## Attribution

All raw data comes from [`doehm/survivoR`](https://github.com/doehm/survivoR)
by Daniel Oehm, MIT licensed. This project is an independent derivative work
and is not affiliated with CBS or Survivor's producers.
