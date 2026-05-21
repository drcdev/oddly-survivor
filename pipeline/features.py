"""Build one-row-per-US-season feature frame.

Each feature has a registry entry (label, description, domain) used both for
the spuriousness score and for the rendered site.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from pipeline.ingest import load_us

VOWELS = set("aeiouAEIOU")
SCRABBLE = {
    **dict.fromkeys("aeioulnstrAEIOULNSTR", 1),
    **dict.fromkeys("dgDG", 2),
    **dict.fromkeys("bcmpBCMP", 3),
    **dict.fromkeys("fhvwyFHVWY", 4),
    **dict.fromkeys("kK", 5),
    **dict.fromkeys("jxJX", 8),
    **dict.fromkeys("qzQZ", 10),
}


@dataclass(frozen=True)
class FeatureMeta:
    key: str
    label: str
    description: str
    domain: str
    source_tables: tuple[str, ...]
    units: str = ""
    higher_is: str = "neutral"


REGISTRY: list[FeatureMeta] = []
_BUILDERS: dict[str, callable] = {}


def feature(label: str, description: str, domain: str, source_tables: tuple[str, ...], units: str = "", higher_is: str = "neutral"):
    def deco(fn):
        key = fn.__name__
        REGISTRY.append(
            FeatureMeta(
                key=key,
                label=label,
                description=description,
                domain=domain,
                source_tables=source_tables,
                units=units,
                higher_is=higher_is,
            )
        )
        _BUILDERS[key] = fn
        return fn

    return deco


def _scrabble_score(name: str) -> int:
    if not isinstance(name, str):
        return 0
    return sum(SCRABBLE.get(c, 0) for c in name)


def _syllables(word: str) -> int:
    if not isinstance(word, str) or not word:
        return 0
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and n > 1:
        n -= 1
    return max(n, 1)


def _gini(values: pd.Series) -> float:
    v = np.asarray(values.dropna(), dtype=float)
    if v.size == 0 or v.sum() == 0:
        return np.nan
    v = np.sort(v)
    n = v.size
    cum = np.cumsum(v)
    return (n + 1 - 2 * np.sum(cum) / cum[-1]) / n


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


# ---- season_summary direct passthroughs --------------------------------------


@feature("Cast size", "Number of castaways at season start", "production", ("season_summary",), "people")
def n_cast(t):
    return t["season_summary"].set_index("season")["n_cast"].astype(float)


@feature("Tribe count", "Number of starting tribes", "production", ("season_summary",), "tribes")
def n_tribes(t):
    return t["season_summary"].set_index("season")["n_tribes"].astype(float)


@feature("Jury size", "Number of jurors at finale", "gameplay", ("season_summary",), "people")
def n_jury(t):
    return t["season_summary"].set_index("season")["n_jury"].astype(float)


@feature("Premiere viewers", "Premiere episode viewers (millions)", "viewership", ("season_summary",), "M")
def premiere_viewers_m(t):
    return (t["season_summary"].set_index("season")["viewers_premiere"] / 1e6).astype(float)


@feature("Finale viewers", "Finale episode viewers (millions)", "viewership", ("season_summary",), "M")
def finale_viewers_m(t):
    return (t["season_summary"].set_index("season")["viewers_finale"] / 1e6).astype(float)


@feature("Mean viewers", "Season-mean episode viewers (millions)", "viewership", ("season_summary",), "M")
def mean_viewers_m(t):
    return (t["season_summary"].set_index("season")["viewers_mean"] / 1e6).astype(float)


@feature("Viewer drop", "Premiere viewers minus finale viewers (millions)", "viewership", ("season_summary",), "M")
def viewer_drop_m(t):
    ss = t["season_summary"].set_index("season")
    return ((ss["viewers_premiere"] - ss["viewers_finale"]) / 1e6).astype(float)


@feature("Premiere month", "Calendar month of season premiere", "production", ("season_summary",), "month")
def premiere_month(t):
    return _to_date(t["season_summary"].set_index("season")["premiered"]).dt.month.astype(float)


@feature("Premiere year", "Calendar year of season premiere", "production", ("season_summary",), "year")
def premiere_year(t):
    return _to_date(t["season_summary"].set_index("season")["premiered"]).dt.year.astype(float)


@feature("Season length", "Days from premiere to finale", "production", ("season_summary",), "days")
def season_length_days(t):
    ss = t["season_summary"].set_index("season")
    return (_to_date(ss["ended"]) - _to_date(ss["premiered"])).dt.days.astype(float)


@feature("Filming length", "Days from filming start to filming end", "production", ("season_summary",), "days")
def filming_length_days(t):
    ss = t["season_summary"].set_index("season")
    return (_to_date(ss["filming_ended"]) - _to_date(ss["filming_started"])).dt.days.astype(float)


@feature("Season name length", "Letter count of season name", "name_text", ("season_summary",), "letters")
def season_name_length(t):
    ss = t["season_summary"].set_index("season")
    return ss["season_name"].fillna("").str.replace(r"[^A-Za-z]", "", regex=True).str.len().astype(float)


# ---- castaways aggregates ----------------------------------------------------


@feature("Mean cast age", "Mean age of all castaways", "demographic", ("castaways",), "years")
def mean_age(t):
    return t["castaways"].groupby("season")["age"].mean().astype(float)


@feature("Median cast age", "Median age of all castaways", "demographic", ("castaways",), "years")
def median_age(t):
    return t["castaways"].groupby("season")["age"].median().astype(float)


@feature("Cast age stddev", "Standard deviation of castaway ages", "demographic", ("castaways",), "years")
def age_stddev(t):
    return t["castaways"].groupby("season")["age"].std().astype(float)


@feature("Oldest castaway", "Age of oldest castaway", "demographic", ("castaways",), "years")
def max_age(t):
    return t["castaways"].groupby("season")["age"].max().astype(float)


@feature("Youngest castaway", "Age of youngest castaway", "demographic", ("castaways",), "years")
def min_age(t):
    return t["castaways"].groupby("season")["age"].min().astype(float)


def _winner_rows(cast: pd.DataFrame) -> pd.DataFrame:
    w = cast.copy()
    w["winner"] = pd.to_numeric(w["winner"], errors="coerce").fillna(0)
    return w[w["winner"] >= 1.0]


@feature("Winner age", "Age of season winner", "demographic", ("castaways",), "years")
def winner_age(t):
    w = _winner_rows(t["castaways"])
    return w.groupby("season")["age"].first().astype(float)


@feature("Winner name length", "Letters in winner's first name", "name_text", ("castaways",), "letters")
def winner_name_length(t):
    w = _winner_rows(t["castaways"])
    first = w.groupby("season")["castaway"].first().fillna("")
    return first.str.replace(r"[^A-Za-z]", "", regex=True).str.len().astype(float)


@feature("Winner name vowel rate", "Vowels divided by letters in winner's first name", "name_text", ("castaways",))
def winner_name_vowel_rate(t):
    w = _winner_rows(t["castaways"])
    first = w.groupby("season")["castaway"].first().fillna("")

    def vrate(s: str) -> float:
        letters = [c for c in s if c.isalpha()]
        return sum(1 for c in letters if c in VOWELS) / len(letters) if letters else np.nan

    return first.map(vrate).astype(float)


@feature("Winner Scrabble score", "Scrabble tile score of winner's first name", "name_text", ("castaways",), "points")
def winner_scrabble(t):
    w = _winner_rows(t["castaways"])
    first = w.groupby("season")["castaway"].first().fillna("")
    return first.map(_scrabble_score).astype(float)


@feature("Mean name syllables", "Mean syllable count of castaway first names", "name_text", ("castaways",), "syllables")
def mean_name_syllables(t):
    cast = t["castaways"]
    return cast.assign(syl=cast["castaway"].fillna("").map(_syllables)).groupby("season")["syl"].mean().astype(float)


@feature("Mean name length", "Mean letter count of castaway first names", "name_text", ("castaways",), "letters")
def mean_name_length(t):
    cast = t["castaways"]
    lens = cast["castaway"].fillna("").str.replace(r"[^A-Za-z]", "", regex=True).str.len()
    return cast.assign(L=lens).groupby("season")["L"].mean().astype(float)


@feature("Vowel-start castaway rate", "Share of castaways whose first name starts with a vowel", "name_text", ("castaways",))
def pct_vowel_start(t):
    cast = t["castaways"]
    starts = cast["castaway"].fillna("").str[:1].str.upper().isin(list("AEIOU"))
    return cast.assign(v=starts).groupby("season")["v"].mean().astype(float)


@feature("West-coast castaway rate", "Share of US castaways from CA/OR/WA", "geographic", ("castaways",))
def pct_west_coast(t):
    cast = t["castaways"]
    west = cast["state"].fillna("").isin(["California", "Oregon", "Washington"])
    return cast.assign(w=west).groupby("season")["w"].mean().astype(float)


@feature("California castaway rate", "Share of castaways from California", "geographic", ("castaways",))
def pct_california(t):
    cast = t["castaways"]
    ca = cast["state"].fillna("") == "California"
    return cast.assign(c=ca).groupby("season")["c"].mean().astype(float)


@feature("Unique home states", "Distinct home states represented in cast", "geographic", ("castaways",), "states")
def n_unique_states(t):
    return t["castaways"].groupby("season")["state"].nunique().astype(float)


# ---- castaway_details joined to castaways ------------------------------------


def _joined_details(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return t["castaways"].merge(
        t["castaway_details"][
            ["castaway_id", "gender", "bipoc", "lgbt", "african", "asian", "latin_american", "native_american"]
        ],
        on="castaway_id",
        how="left",
    )


@feature("Female castaway rate", "Share of castaways identifying as female", "demographic", ("castaways", "castaway_details"))
def pct_female(t):
    j = _joined_details(t)
    f = (j["gender"].fillna("") == "Female").astype(float)
    return j.assign(f=f).groupby("season")["f"].mean().astype(float)


@feature("BIPOC castaway rate", "Share of castaways identifying as BIPOC", "demographic", ("castaways", "castaway_details"))
def pct_bipoc(t):
    j = _joined_details(t)
    b = pd.to_numeric(j["bipoc"], errors="coerce").fillna(0)
    return j.assign(b=b).groupby("season")["b"].mean().astype(float)


@feature("LGBT castaway rate", "Share of castaways identifying as LGBT", "demographic", ("castaways", "castaway_details"))
def pct_lgbt(t):
    j = _joined_details(t)
    v = pd.to_numeric(j["lgbt"], errors="coerce").fillna(0)
    return j.assign(v=v).groupby("season")["v"].mean().astype(float)


# ---- episodes ----------------------------------------------------------------


@feature("Episode count", "Number of aired episodes", "production", ("episodes",), "episodes")
def n_episodes(t):
    return t["episodes"].groupby("season")["episode"].nunique().astype(float)


@feature("Mean episode rating", "Mean IMDb rating across episodes", "viewership", ("episodes",), "stars")
def mean_imdb(t):
    return t["episodes"].groupby("season")["imdb_rating"].mean().astype(float)


@feature("Top-rated episode", "Maximum IMDb episode rating", "viewership", ("episodes",), "stars")
def max_imdb(t):
    return t["episodes"].groupby("season")["imdb_rating"].max().astype(float)


@feature("Episode rating stddev", "Standard deviation of IMDb episode ratings", "viewership", ("episodes",), "stars")
def stddev_imdb(t):
    return t["episodes"].groupby("season")["imdb_rating"].std().astype(float)


@feature("Mean episode length", "Mean episode runtime (minutes)", "production", ("episodes",), "min")
def mean_ep_length(t):
    return t["episodes"].groupby("season")["episode_length"].mean().astype(float)


# ---- vote_history ------------------------------------------------------------


@feature("Tribal council count", "Distinct tribal councils per season", "gameplay", ("vote_history",), "councils")
def n_tribal_councils(t):
    vh = t["vote_history"]
    return vh.groupby("season")["episode"].nunique().astype(float)


@feature("Total votes cast", "Total individual votes cast at all tribals", "gameplay", ("vote_history",), "votes")
def n_votes_total(t):
    return t["vote_history"].groupby("season").size().astype(float)


@feature("Tied vote rate", "Share of vote rows flagged as a tie", "gameplay", ("vote_history",))
def tie_rate(t):
    vh = t["vote_history"].copy()
    vh["tie"] = pd.to_numeric(vh["tie"], errors="coerce").fillna(0)
    return vh.groupby("season")["tie"].mean().astype(float)


@feature("Nullified vote rate", "Share of votes nullified by an idol or advantage", "gameplay", ("vote_history",))
def nullified_rate(t):
    vh = t["vote_history"].copy()
    vh["nullified"] = pd.to_numeric(vh["nullified"], errors="coerce").fillna(0)
    return vh.groupby("season")["nullified"].mean().astype(float)


# ---- jury_votes --------------------------------------------------------------


@feature("Winner jury share", "Share of jury votes the winner received", "gameplay", ("jury_votes", "castaways"))
def winner_jury_share(t):
    jv = t["jury_votes"]
    cast = t["castaways"].copy()
    cast["winner"] = pd.to_numeric(cast["winner"], errors="coerce").fillna(0)
    winners = cast[cast["winner"] >= 1.0][["season", "castaway_id"]].rename(columns={"castaway_id": "winner_id"})
    merged = jv.merge(winners, on="season", how="left")
    merged["is_winner"] = (merged["finalist_id"] == merged["winner_id"]).astype(float)
    grouped = merged.groupby("season").agg(
        total=("is_winner", "size"),
        for_winner=("is_winner", "sum"),
    )
    return (grouped["for_winner"] / grouped["total"]).astype(float)


# ---- challenge_description + challenge_results --------------------------------


@feature("Endurance challenge rate", "Share of challenges tagged as endurance", "challenge", ("challenge_description",))
def pct_endurance_challenges(t):
    cd = t["challenge_description"].copy()
    cd["endurance"] = pd.to_numeric(cd["endurance"], errors="coerce").fillna(0)
    return cd.groupby("season")["endurance"].mean().astype(float)


@feature("Puzzle challenge rate", "Share of challenges tagged as puzzles", "challenge", ("challenge_description",))
def pct_puzzle_challenges(t):
    cd = t["challenge_description"].copy()
    cd["puzzle"] = pd.to_numeric(cd["puzzle"], errors="coerce").fillna(0)
    return cd.groupby("season")["puzzle"].mean().astype(float)


@feature("Water challenge rate", "Share of challenges tagged as water-based", "challenge", ("challenge_description",))
def pct_water_challenges(t):
    cd = t["challenge_description"].copy()
    cd["water"] = pd.to_numeric(cd["water"], errors="coerce").fillna(0)
    return cd.groupby("season")["water"].mean().astype(float)


@feature("Strength challenge rate", "Share of challenges tagged as strength", "challenge", ("challenge_description",))
def pct_strength_challenges(t):
    cd = t["challenge_description"].copy()
    cd["strength"] = pd.to_numeric(cd["strength"], errors="coerce").fillna(0)
    return cd.groupby("season")["strength"].mean().astype(float)


@feature("Winner immunity wins", "Individual immunities won by season winner", "challenge", ("challenge_results", "castaways"), "wins")
def winner_immunity_wins(t):
    cast = t["castaways"].copy()
    cast["winner"] = pd.to_numeric(cast["winner"], errors="coerce").fillna(0)
    winners = cast[cast["winner"] >= 1.0][["season", "castaway_id"]]
    cr = t["challenge_results"].copy()
    cr["won_individual_immunity"] = pd.to_numeric(cr["won_individual_immunity"], errors="coerce").fillna(0)
    merged = cr.merge(winners, on=["season", "castaway_id"], how="inner")
    return merged.groupby("season")["won_individual_immunity"].sum().astype(float)


# ---- confessionals -----------------------------------------------------------


@feature("Total confessionals", "Total confessional clips across season", "confessional", ("confessionals",), "clips")
def total_confessionals(t):
    cf = t["confessionals"].copy()
    cf["confessional_count"] = pd.to_numeric(cf["confessional_count"], errors="coerce").fillna(0)
    return cf.groupby("season")["confessional_count"].sum().astype(float)


@feature("Winner confessional share", "Share of confessional clips by the season winner", "confessional", ("confessionals", "castaways"))
def winner_confessional_share(t):
    cast = t["castaways"].copy()
    cast["winner"] = pd.to_numeric(cast["winner"], errors="coerce").fillna(0)
    winners = cast[cast["winner"] >= 1.0][["season", "castaway_id"]]
    cf = t["confessionals"].copy()
    cf["confessional_count"] = pd.to_numeric(cf["confessional_count"], errors="coerce").fillna(0)
    total = cf.groupby("season")["confessional_count"].sum()
    winner_cf = cf.merge(winners, on=["season", "castaway_id"], how="inner").groupby("season")["confessional_count"].sum()
    return (winner_cf / total).astype(float)


@feature("Confessional inequality", "Gini index of confessional counts per castaway", "confessional", ("confessionals",))
def confessional_gini(t):
    cf = t["confessionals"].copy()
    cf["confessional_count"] = pd.to_numeric(cf["confessional_count"], errors="coerce").fillna(0)
    by_castaway = cf.groupby(["season", "castaway_id"])["confessional_count"].sum().reset_index()
    return by_castaway.groupby("season")["confessional_count"].apply(_gini).astype(float)


# ---- advantage_movement ------------------------------------------------------


@feature("Idol/advantage events", "Total advantage-movement events (find, play, transfer)", "gameplay", ("advantage_movement",), "events")
def n_advantage_events(t):
    return t["advantage_movement"].groupby("season").size().astype(float).reindex(range(1, 51)).fillna(0)


@feature("Idols played", "Count of 'play' events on hidden idols/advantages", "gameplay", ("advantage_movement",), "plays")
def n_idols_played(t):
    am = t["advantage_movement"].copy()
    am["event"] = am["event"].fillna("").astype(str).str.lower()
    return am[am["event"].str.contains("play")].groupby("season").size().astype(float).reindex(range(1, 51)).fillna(0)


# ---- tribe naming ------------------------------------------------------------


@feature("Mean tribe name length", "Mean letter count of unique tribe names", "name_text", ("tribe_mapping",), "letters")
def mean_tribe_name_length(t):
    tm = t["tribe_mapping"]
    tribes = tm.dropna(subset=["tribe"]).groupby("season")["tribe"].unique()
    return tribes.apply(lambda arr: float(np.mean([len(str(x)) for x in arr]))).astype(float)


@feature("Tribes starting with vowel", "Share of distinct tribe names starting with a vowel", "name_text", ("tribe_mapping",))
def pct_vowel_tribes(t):
    tm = t["tribe_mapping"]
    tribes = tm.dropna(subset=["tribe"]).groupby("season")["tribe"].unique()
    return tribes.apply(lambda arr: float(np.mean([str(x)[:1].upper() in VOWELS for x in arr]))).astype(float)


# ---- boot_mapping ------------------------------------------------------------


@feature("Premerge female boot rate", "Share of pre-merge boots who were female", "gameplay", ("castaways", "castaway_details"))
def pct_premerge_female_boots(t):
    cast = t["castaways"].copy()
    cast["jury"] = cast["jury"].astype(str).str.lower().isin(["true", "1", "1.0"])
    cast["finalist"] = cast["finalist"].astype(str).str.lower().isin(["true", "1", "1.0"])
    premerge = cast[~cast["jury"] & ~cast["finalist"]]
    joined = premerge.merge(t["castaway_details"][["castaway_id", "gender"]], on="castaway_id", how="left")
    joined["is_female"] = (joined["gender"].fillna("") == "Female").astype(float)
    return joined.groupby("season")["is_female"].mean().astype(float)


@feature("First boot age", "Age of the season's first eliminated castaway", "demographic", ("castaways",), "years")
def first_boot_age(t):
    cast = t["castaways"]
    # the row with the largest 'place' is the first boot
    return cast.sort_values(["season", "place"], ascending=[True, False]).groupby("season").first()["age"].astype(float)


# ---- driver ------------------------------------------------------------------


def build(tag: str) -> pd.DataFrame:
    tables = {
        name: load_us(name, tag)
        for name in [
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
        ]
    }
    seasons = pd.Index(range(1, 51), name="season")
    out = pd.DataFrame(index=seasons)
    for meta in REGISTRY:
        try:
            series = _BUILDERS[meta.key](tables)
            series.index = pd.Index(series.index.astype("int64"), name="season")
            out[meta.key] = series.reindex(seasons)
        except Exception as e:
            raise RuntimeError(f"feature {meta.key!r} failed: {e}") from e
    return out


def registry_dict() -> list[dict]:
    return [asdict(m) for m in REGISTRY]
