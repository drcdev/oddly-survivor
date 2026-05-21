"""Render correlation charts as inline SVG + Open Graph PNG."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class ChartSpec:
    title_a: str
    title_b: str
    color_a: str
    color_b: str
    units_a: str
    units_b: str


def _setup_axes(ax, color: str, label: str, units: str):
    ax.tick_params(axis="y", colors=color, labelsize=8)
    ax.spines["right"].set_color(color)
    ax.spines["left"].set_color(color)
    ylabel = f"{label}" + (f" ({units})" if units else "")
    ax.set_ylabel(ylabel, color=color, fontsize=9)


def _draw(seasons: list[int], a_vals: list[float], b_vals: list[float], spec: ChartSpec, figsize, dpi):
    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    ax2 = ax1.twinx()

    ax1.plot(seasons, a_vals, color=spec.color_a, marker="o", markersize=4, linewidth=1.8, label=spec.title_a)
    ax2.plot(seasons, b_vals, color=spec.color_b, marker="s", markersize=4, linewidth=1.8, linestyle="--", label=spec.title_b)

    ax1.set_xlabel("Season", fontsize=9)
    _setup_axes(ax1, spec.color_a, spec.title_a, spec.units_a)
    _setup_axes(ax2, spec.color_b, spec.title_b, spec.units_b)

    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.grid(axis="y", alpha=0.15)
    ax1.set_xlim(0, 51)

    fig.tight_layout()
    return fig


def render_svg(seasons: pd.Index, series_a: pd.Series, series_b: pd.Series, spec: ChartSpec) -> str:
    s = list(seasons)
    a = [None if pd.isna(v) else float(v) for v in series_a.tolist()]
    b = [None if pd.isna(v) else float(v) for v in series_b.tolist()]
    fig = _draw(s, a, b, spec, figsize=(7.5, 3.6), dpi=110)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    # strip the XML prolog so we can inline directly into HTML
    svg = re.sub(r"^<\?xml[^>]+\?>\s*", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]+>\s*", "", svg)
    return svg


def render_png(seasons: pd.Index, series_a: pd.Series, series_b: pd.Series, spec: ChartSpec, dest_path: str) -> None:
    s = list(seasons)
    a = [None if pd.isna(v) else float(v) for v in series_a.tolist()]
    b = [None if pd.isna(v) else float(v) for v in series_b.tolist()]
    fig = _draw(s, a, b, spec, figsize=(12, 6.3), dpi=100)
    fig.savefig(dest_path, format="png", bbox_inches="tight")
    plt.close(fig)


def render_thumbnail_svg(seasons: pd.Index, series_a: pd.Series, series_b: pd.Series, spec: ChartSpec) -> str:
    s = list(seasons)
    a = [None if pd.isna(v) else float(v) for v in series_a.tolist()]
    b = [None if pd.isna(v) else float(v) for v in series_b.tolist()]
    fig, ax1 = plt.subplots(figsize=(4.2, 2.0), dpi=100)
    ax2 = ax1.twinx()
    ax1.plot(s, a, color=spec.color_a, linewidth=1.3)
    ax2.plot(s, b, color=spec.color_b, linewidth=1.3, linestyle="--")
    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax1.set_xlim(0, 51)
    fig.tight_layout(pad=0.1)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    svg = buf.getvalue()
    svg = re.sub(r"^<\?xml[^>]+\?>\s*", "", svg)
    svg = re.sub(r"<!DOCTYPE[^>]+>\s*", "", svg)
    return svg
