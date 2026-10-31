#!/usr/bin/env python3
"""Offline evaluation of the saved Week 3 Jacobian-lens bias readouts.

This script reads ONLY the already-saved public CSVs in ``Week 3/results/``:

* ``week3_raw_results.csv``
* ``week3_summary_by_layer_format.csv``
* ``week3_summary_by_layer_condition_format.csv``
* ``week3_summary_by_layer_polarity_format.csv``

It loads no model, runs no inference, uses no GPU/network, and never writes to
the source CSVs (their SHA-256 hashes are checked before and after). All
generated files are written next to this script:

* ``offline_evaluation_results.csv``  - concise long-format results table
* ``fig_layerwise_format_gaps.png``   - per-layer matched MC/open comparison
* ``fig_lens_vs_output_and_pairs.png`` - L15-L22 lens vs output layer, and pairs
* ``offline_evaluation_report.md``    - interpretation and limitations

Sign convention (from Week 3): ``rank_gap = counter_rank - stereotype_rank``.
A negative gap means the counter-stereotype target ranks stronger (smaller
rank) than the stereotype target. Token ranks are NOT generated answers; this
script never infers what the model would have generated.

Usage::

    python evaluate_saved_week3.py [--repo-root PATH] [--n-boot N] [--seed S]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
# <repo>/Week 4/submissions/MelodyResearch_offline_evaluation/evaluate_saved_week3.py
DEFAULT_REPO_ROOT = SCRIPT_DIR.parents[2]

RAW_NAME = "week3_raw_results.csv"
SUMMARY_SPECS = {
    "week3_summary_by_layer_format.csv": ["layer", "prompt_format"],
    "week3_summary_by_layer_condition_format.csv": ["layer", "context_condition", "prompt_format"],
    "week3_summary_by_layer_polarity_format.csv": ["layer", "question_polarity", "prompt_format"],
}
RAW_COLUMNS = [
    "category", "base_item_id", "example_id", "pair_id", "context_condition",
    "question_polarity", "prompt_format", "prompt", "question", "layer", "readout",
    "seq_len", "analysis_position", "source_token", "top20_tokens", "top20_ranks",
    "stereotype_group", "counter_group", "stereotype_target_text", "counter_target_text",
    "stereotype_constituent_tokens", "stereotype_constituent_ranks",
    "counter_constituent_tokens", "counter_constituent_ranks", "stereotype_rank",
    "counter_rank", "rank_gap", "unknown_answer_key", "correct_answer_label",
    "stereotype_answer_key", "counter_answer_key",
]

LENS = "jacobian_lens"
OUTPUT = "model_output"
MC = "multiple_choice"
OPEN = "open_ended"
FORMATS = (MC, OPEN)
CONTEXTS = ("ambig", "disambig")
POLARITIES = ("neg", "nonneg")
LENS_LAYERS = list(range(23))
OUTPUT_LAYER = 23
LATE_LO, LATE_HI = 15, 22
LATE_LABEL = f"L{LATE_LO}-L{LATE_HI}"

N_RAW_ROWS = 1920
N_BASE_ITEMS = 10
N_EVALS = 80
N_PAIRS = 40
ROWS_PER_EVAL = 24

BASE_KEY = ["category", "base_item_id"]
PAIR_KEY = BASE_KEY + ["context_condition", "question_polarity"]
EVAL_KEY = PAIR_KEY + ["prompt_format"]
READOUT_KEY = EVAL_KEY + ["readout", "layer"]

# Existing Week 3 reference results. Used ONLY as validation assertions against
# the values recomputed below; reported numbers always come from the data.
REF_LATE_MC_MEAN = -24761.128125
REF_LATE_OPEN_MEAN = -897.165625
REF_OPEN_SMALLER = 37
REF_SIGN_PRESERVED = 40
REF_TOL = 1e-6

# Summary CSVs may have been written with rounded floats; means of integer
# ranks over 20 or 40 rows have at most 3 decimals, so 0.01 is still strict.
SUMMARY_ATOL = 0.01

SYMLOG_LINTHRESH = 10.0


class ValidationError(RuntimeError):
    """Raised when the saved Week 3 data fail a validation check."""


def check(cond: bool, msg: str, log: list[str]) -> None:
    if not cond:
        raise ValidationError(f"FAILED: {msg}")
    log.append(msg)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValidationError(f"FAILED: raw CSV missing columns {missing}")
    for col in ("stereotype_rank", "counter_rank", "rank_gap"):
        df[col] = pd.to_numeric(df[col], errors="raise").astype(float)
    layer = pd.to_numeric(df["layer"], errors="raise").astype(float)
    if not np.all(np.isfinite(layer)) or not np.all(layer == np.round(layer)):
        raise ValidationError("FAILED: non-integer layer values in raw CSV")
    df["layer"] = layer.astype(int)
    return df


def validate_raw(df: pd.DataFrame, log: list[str]) -> None:
    check(len(df) == N_RAW_ROWS, f"raw CSV has exactly {N_RAW_ROWS:,} rows (found {len(df):,})", log)
    check(set(df["readout"]) == {LENS, OUTPUT}, f"readouts are exactly {{{LENS}, {OUTPUT}}}", log)
    check(set(df["prompt_format"]) == set(FORMATS), f"prompt formats are exactly {set(FORMATS)}", log)
    check(set(df["context_condition"]) == set(CONTEXTS), f"contexts are exactly {set(CONTEXTS)}", log)
    check(set(df["question_polarity"]) == set(POLARITIES), f"polarities are exactly {set(POLARITIES)}", log)
    for col in ("stereotype_rank", "counter_rank", "rank_gap"):
        check(bool(np.all(np.isfinite(df[col]))), f"`{col}` is finite in every row", log)

    n_base = len(df[BASE_KEY].drop_duplicates())
    check(n_base == N_BASE_ITEMS, f"{N_BASE_ITEMS} unique base items (category, base_item_id) (found {n_base})", log)
    evals = df.groupby(EVAL_KEY).size()
    check(len(evals) == N_EVALS, f"{N_EVALS} unique item/condition/polarity/format evaluations (found {len(evals)})", log)
    check(bool((evals == ROWS_PER_EVAL).all()), f"every evaluation has exactly {ROWS_PER_EVAL} readout rows", log)
    evals_per_base = df[EVAL_KEY].drop_duplicates().groupby(BASE_KEY).size()
    check(bool((evals_per_base == 8).all()), "every base item has 2 contexts x 2 polarities x 2 formats = 8 evaluations", log)

    n_dup = int(df.duplicated(READOUT_KEY).sum())
    check(n_dup == 0, f"no duplicate readout keys {READOUT_KEY} (found {n_dup})", log)

    lens = df[df["readout"] == LENS]
    out = df[df["readout"] == OUTPUT]
    lens_ok = lens.groupby(EVAL_KEY)["layer"].apply(lambda s: sorted(s.tolist()) == LENS_LAYERS)
    check(len(lens_ok) == N_EVALS and bool(lens_ok.all()),
          f"every evaluation has {LENS} rows at exactly layers 0-22", log)
    out_counts = out.groupby(EVAL_KEY).size()
    check(len(out_counts) == N_EVALS and bool((out_counts == 1).all()),
          f"every evaluation has exactly one {OUTPUT} row", log)
    check(bool((out["layer"] == OUTPUT_LAYER).all()), f"every {OUTPUT} row is at layer {OUTPUT_LAYER}", log)

    diff = (df["rank_gap"] - (df["counter_rank"] - df["stereotype_rank"])).abs()
    check(bool((diff <= 1e-9).all()),
          f"rank_gap == counter_rank - stereotype_rank exactly in all rows (max |diff| = {diff.max():g})", log)

    fmt_per_pair = df.groupby(PAIR_KEY)["prompt_format"].nunique()
    check(len(fmt_per_pair) == N_PAIRS and bool((fmt_per_pair == 2).all()),
          f"{N_PAIRS} matched format pairs, each with both formats", log)
    targets = df.groupby(PAIR_KEY)[["stereotype_target_text", "counter_target_text"]].nunique()
    check(bool((targets == 1).all().all()),
          "stereotype/counter target texts are identical across formats, layers and readouts within each pair", log)


def read_summary_csv(path: Path) -> pd.DataFrame:
    """Read a summary CSV that may have one or several header rows."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.reader(fh) if any(c.strip() for c in r)]
    first_data = next((i for i, r in enumerate(rows) if any(c.strip() in FORMATS for c in r)), None)
    if not first_data:
        raise ValidationError(f"FAILED: could not locate header/data rows in {path.name}")
    header_rows = rows[:first_data]
    width = max(len(r) for r in rows)
    names = []
    for j in range(width):
        parts = [r[j].strip() for r in header_rows
                 if j < len(r) and r[j].strip() and not r[j].strip().startswith("Unnamed")]
        names.append("_".join(parts) if parts else f"col{j}")
    if len(set(names)) != len(names):
        raise ValidationError(f"FAILED: ambiguous column names in {path.name}: {names}")
    data = [r + [""] * (width - len(r)) for r in rows[first_data:]]
    return pd.DataFrame(data, columns=names)


def identify_summary_columns(sdf: pd.DataFrame, group_cols: list[str], fname: str) -> dict[str, str]:
    known = {"prompt_format": set(FORMATS), "context_condition": set(CONTEXTS),
             "question_polarity": set(POLARITIES)}
    lowered = {c: c.lower() for c in sdf.columns}
    values = {c: set(sdf[c].str.strip()) for c in sdf.columns}
    mapping: dict[str, str] = {}
    for g in group_cols:
        if g == "layer":
            cands = [c for c in sdf.columns if "layer" in lowered[c]]
        else:
            cands = [c for c in sdf.columns if values[c] and values[c] <= known[g]]
        if len(cands) != 1:
            raise ValidationError(f"FAILED: cannot identify `{g}` column in {fname} (candidates {cands})")
        mapping[g] = cands[0]
    readout_cands = [c for c in sdf.columns if values[c] and values[c] <= {LENS, OUTPUT}]
    if readout_cands:
        mapping["readout"] = readout_cands[0]
    used = set(mapping.values())
    for stat in ("mean", "median", "count"):
        cands = [c for c in sdf.columns if stat in lowered[c] and c not in used]
        if not cands and stat == "count":
            cands = [c for c in sdf.columns if lowered[c] in ("n", "n_rows", "rank_gap_n") and c not in used]
        if len(cands) > 1:
            cands = [c for c in cands if "gap" in lowered[c]]
        if len(cands) != 1:
            raise ValidationError(f"FAILED: cannot identify rank_gap `{stat}` column in {fname} (candidates {cands})")
        mapping[stat] = cands[0]
    return mapping


def validate_summary(path: Path, group_cols: list[str], lens: pd.DataFrame, log: list[str]) -> dict:
    sdf = read_summary_csv(path)
    cols = identify_summary_columns(sdf, group_cols, path.name)
    if "readout" in cols:
        check(set(sdf[cols["readout"]].str.strip()) == {LENS},
              f"{path.name}: summary covers only {LENS} rows", log)
    norm = pd.DataFrame({g: sdf[cols[g]].str.strip() for g in group_cols if g != "layer"})
    layer = pd.to_numeric(sdf[cols["layer"]], errors="raise").astype(float)
    if not np.all(layer == np.round(layer)):
        raise ValidationError(f"FAILED: non-integer layers in {path.name}")
    norm["layer"] = layer.astype(int).to_numpy()
    norm["mean"] = pd.to_numeric(sdf[cols["mean"]], errors="raise").astype(float).to_numpy()
    norm["median"] = pd.to_numeric(sdf[cols["median"]], errors="raise").astype(float).to_numpy()
    norm["count"] = pd.to_numeric(sdf[cols["count"]], errors="raise").astype(float).to_numpy()

    check(sorted(norm["layer"].unique().tolist()) == LENS_LAYERS,
          f"{path.name}: covers {LENS} layers 0-22 only", log)
    check(int(norm.duplicated(group_cols).sum()) == 0, f"{path.name}: no duplicate group keys", log)

    rec = lens.groupby(group_cols)["rank_gap"].agg(["mean", "median", "count"]).reset_index()
    merged = rec.merge(norm, on=group_cols, how="outer", suffixes=("_rec", "_src"), indicator=True)
    check(bool((merged["_merge"] == "both").all()) and len(merged) == len(rec) == len(norm),
          f"{path.name}: same {len(rec)} groups as recomputed from raw lens rows", log)
    d_mean = float((merged["mean_rec"] - merged["mean_src"]).abs().max())
    d_median = float((merged["median_rec"] - merged["median_src"]).abs().max())
    tol_mean = SUMMARY_ATOL + 1e-9 * merged["mean_rec"].abs()
    tol_median = SUMMARY_ATOL + 1e-9 * merged["median_rec"].abs()
    check(bool(((merged["mean_rec"] - merged["mean_src"]).abs() <= tol_mean).all()),
          f"{path.name}: rank_gap means match recomputation (max |diff| = {d_mean:.3g})", log)
    check(bool(((merged["median_rec"] - merged["median_src"]).abs() <= tol_median).all()),
          f"{path.name}: rank_gap medians match recomputation (max |diff| = {d_median:.3g})", log)
    check(bool((merged["count_rec"] == merged["count_src"]).all()),
          f"{path.name}: rank_gap counts match recomputation exactly", log)
    return {"n_groups": len(rec), "max_abs_diff_mean": d_mean, "max_abs_diff_median": d_median,
            "counts": sorted(set(rec["count"].astype(int).tolist()))}


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def sgn(x) -> np.ndarray:
    return np.sign(np.asarray(x, dtype=float))


def trimmed_mean(x, prop: float = 0.1) -> float:
    arr = np.sort(np.asarray(x, dtype=float))
    k = int(math.floor(prop * len(arr)))
    return float(arr[k:len(arr) - k].mean())


def spearman(a, b) -> float:
    a = pd.Series(np.asarray(a, dtype=float)).rank()
    b = pd.Series(np.asarray(b, dtype=float)).rank()
    return float(a.corr(b))


def base_label(row) -> str:
    return f"{row['category']}:{row['base_item_id']}"


def cluster_bootstrap(per_item: pd.DataFrame, specs: dict, n_boot: int, seed: int) -> dict:
    """Deterministic base-item (cluster) bootstrap of ratio statistics."""
    rng = np.random.default_rng(seed)
    k = len(per_item)
    idx = rng.integers(0, k, size=(n_boot, k))
    weights = np.zeros((n_boot, k))
    np.add.at(weights, (np.repeat(np.arange(n_boot), k), idx.ravel()), 1.0)
    out = {}
    for name, (num, den) in specs.items():
        nums = per_item[num].to_numpy(float)
        dens = per_item[den].to_numpy(float)
        est = nums.sum() / dens.sum()
        boots = (weights @ nums) / (weights @ dens)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        lobo = (nums.sum() - nums) / (dens.sum() - dens)
        out[name] = {"estimate": float(est), "lo": float(lo), "hi": float(hi),
                     "lobo_min": float(lobo.min()), "lobo_max": float(lobo.max())}
    return out


class Results:
    COLUMNS = ["section", "metric", "unit", "subset", "prompt_format", "layer_range", "value",
               "numerator", "denominator", "n_base_items", "interval_low", "interval_high",
               "interval_method", "note"]

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, section, metric, value, unit, prompt_format="", subset="all", layer_range=LATE_LABEL,
            numerator=None, denominator=None, n_base_items=N_BASE_ITEMS, lo=None, hi=None,
            method="", note="") -> None:
        self.rows.append(dict(section=section, metric=metric, unit=unit, subset=subset,
                              prompt_format=prompt_format, layer_range=layer_range, value=value,
                              numerator=numerator, denominator=denominator, n_base_items=n_base_items,
                              interval_low=lo, interval_high=hi, interval_method=method, note=note))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.COLUMNS)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def analyse(lens: pd.DataFrame, out: pd.DataFrame, log: list[str], n_boot: int, seed: int) -> dict:
    A: dict = {}

    # Per-layer matched MC/open gaps (pair-level: one row per pair per layer).
    lw = (lens.set_index(PAIR_KEY + ["layer", "prompt_format"])["rank_gap"]
          .unstack("prompt_format").reset_index())
    per_layer = []
    for layer, g in lw.groupby("layer"):
        mc = g[MC].to_numpy(float)
        op = g[OPEN].to_numpy(float)
        per_layer.append(dict(
            layer=int(layer), n_pairs=len(g),
            mc_mean=mc.mean(), open_mean=op.mean(),
            mc_median_abs=float(np.median(np.abs(mc))), open_median_abs=float(np.median(np.abs(op))),
            mc_q25_abs=float(np.percentile(np.abs(mc), 25)), mc_q75_abs=float(np.percentile(np.abs(mc), 75)),
            open_q25_abs=float(np.percentile(np.abs(op), 25)), open_q75_abs=float(np.percentile(np.abs(op), 75)),
            open_smaller_abs=int((np.abs(op) < np.abs(mc)).sum()),
            sign_agree=int((sgn(op) == sgn(mc)).sum()),
            mc_negative=int((mc < 0).sum()), open_negative=int((op < 0).sum()),
        ))
    per_layer = pd.DataFrame(per_layer)
    check(len(per_layer) == len(LENS_LAYERS) and bool((per_layer["n_pairs"] == N_PAIRS).all()),
          f"per-layer matched table has {N_PAIRS} pairs at each of 23 lens layers", log)
    A["per_layer"] = per_layer

    # L15-L22 window.
    late = lens[(lens["layer"] >= LATE_LO) & (lens["layer"] <= LATE_HI)]
    late_counts = late.groupby(EVAL_KEY).size()
    check(len(late_counts) == N_EVALS and bool((late_counts == 8).all()),
          f"every evaluation has 8 {LENS} rows in {LATE_LABEL}", log)
    row_stats = late.groupby("prompt_format")["rank_gap"].agg(["mean", "median", "count"])
    A["late_row_stats"] = row_stats

    # Constant sign across the 8 late layers within an evaluation.
    late_nsign = late.assign(s=sgn(late["rank_gap"])).groupby(EVAL_KEY)["s"].nunique().reset_index()
    A["late_const_sign"] = {f: int((late_nsign.loc[late_nsign["prompt_format"] == f, "s"] == 1).sum())
                            for f in FORMATS}

    # Evaluation level: late lens average vs saved output-layer gap, same targets.
    eval_late = late.groupby(EVAL_KEY)["rank_gap"].mean().rename("lens_late_avg")
    lens_targets = late.groupby(EVAL_KEY)[["stereotype_target_text", "counter_target_text"]].first()
    out_idx = out.set_index(EVAL_KEY)
    check(bool((lens_targets.sort_index() ==
                out_idx[["stereotype_target_text", "counter_target_text"]].sort_index()).all().all()),
          f"{OUTPUT} rows use the same target texts as the {LENS} rows of the same evaluation", log)
    ev = pd.concat([eval_late, out_idx["rank_gap"].rename("output_gap")], axis=1).reset_index()
    check(len(ev) == N_EVALS and not ev[["lens_late_avg", "output_gap"]].isna().any().any(),
          f"{N_EVALS} evaluations have both a {LATE_LABEL} lens average and an output-layer gap", log)
    ev["sign_persists"] = sgn(ev["lens_late_avg"]) == sgn(ev["output_gap"])
    ev["output_abs_smaller"] = ev["output_gap"].abs() < ev["lens_late_avg"].abs()
    A["ev"] = ev

    # Pair level (one row per matched MC/open pair).
    pw = ev.set_index(PAIR_KEY + ["prompt_format"])[["lens_late_avg", "output_gap"]].unstack("prompt_format")
    pw.columns = [f"{v}__{f}" for v, f in pw.columns]
    pw = pw.reset_index().rename(columns={
        f"lens_late_avg__{MC}": "mc", f"lens_late_avg__{OPEN}": "open",
        f"output_gap__{MC}": "out_mc", f"output_gap__{OPEN}": "out_open"})
    check(len(pw) == N_PAIRS, f"{N_PAIRS} pair-level rows", log)
    pw["abs_mc"] = pw["mc"].abs()
    pw["abs_open"] = pw["open"].abs()
    pw["delta_abs"] = pw["abs_open"] - pw["abs_mc"]
    pw["log10_abs_ratio"] = np.log10((pw["abs_open"] + 1.0) / (pw["abs_mc"] + 1.0))
    pw["open_smaller"] = (pw["abs_open"] < pw["abs_mc"]).astype(int)
    pw["sign_preserved"] = (sgn(pw["mc"]) == sgn(pw["open"])).astype(int)
    pw["persist_mc"] = (sgn(pw["mc"]) == sgn(pw["out_mc"])).astype(int)
    pw["persist_open"] = (sgn(pw["open"]) == sgn(pw["out_open"])).astype(int)
    pw["out_open_smaller"] = (pw["out_open"].abs() < pw["out_mc"].abs()).astype(int)
    pw["out_sign_preserved"] = (sgn(pw["out_mc"]) == sgn(pw["out_open"])).astype(int)
    pw["one"] = 1
    pw["base_item"] = pw.apply(base_label, axis=1)
    A["pw"] = pw

    # Reference assertions (validation only; reported values come from the data).
    check(abs(row_stats.loc[MC, "mean"] - REF_LATE_MC_MEAN) <= REF_TOL,
          f"reference check: {LATE_LABEL} MC row mean reproduces the Week 3 value", log)
    check(abs(row_stats.loc[OPEN, "mean"] - REF_LATE_OPEN_MEAN) <= REF_TOL,
          f"reference check: {LATE_LABEL} open-ended row mean reproduces the Week 3 value", log)
    check(int(pw["open_smaller"].sum()) == REF_OPEN_SMALLER,
          f"reference check: open-ended |pair average| smaller in the Week 3 number of pairs (of {N_PAIRS})", log)
    check(int(pw["sign_preserved"].sum()) == REF_SIGN_PRESERVED,
          f"reference check: pair-average sign preserved across formats in the Week 3 number of pairs (of {N_PAIRS})", log)

    # Base-item level aggregation.
    per_item = pw.groupby("base_item").agg(
        n_pairs=("one", "sum"), sum_mc=("mc", "sum"), sum_open=("open", "sum"),
        sum_abs_mc=("abs_mc", "sum"), sum_abs_open=("abs_open", "sum"),
        sum_delta_abs=("delta_abs", "sum"), sum_log_ratio=("log10_abs_ratio", "sum"),
        n_open_smaller=("open_smaller", "sum"), n_sign_preserved=("sign_preserved", "sum"),
        n_persist_mc=("persist_mc", "sum"), n_persist_open=("persist_open", "sum"),
        n_out_open_smaller=("out_open_smaller", "sum"),
    )
    check(len(per_item) == N_BASE_ITEMS and bool((per_item["n_pairs"] == 4).all()),
          f"each of the {N_BASE_ITEMS} base items contributes exactly 4 matched pairs", log)
    per_item["mean_mc"] = per_item["sum_mc"] / per_item["n_pairs"]
    per_item["mean_open"] = per_item["sum_open"] / per_item["n_pairs"]
    per_item["share_abs_mc"] = per_item["sum_abs_mc"] / per_item["sum_abs_mc"].sum()
    A["per_item"] = per_item

    specs = {
        "mean_pair_avg_gap_mc": ("sum_mc", "n_pairs"),
        "mean_pair_avg_gap_open": ("sum_open", "n_pairs"),
        "mean_abs_pair_avg_mc": ("sum_abs_mc", "n_pairs"),
        "mean_abs_pair_avg_open": ("sum_abs_open", "n_pairs"),
        "mean_delta_abs_open_minus_mc": ("sum_delta_abs", "n_pairs"),
        "mean_log10_abs_ratio_open_over_mc": ("sum_log_ratio", "n_pairs"),
        "frac_pairs_open_abs_smaller": ("n_open_smaller", "n_pairs"),
        "frac_pairs_sign_preserved": ("n_sign_preserved", "n_pairs"),
        "frac_lens_output_sign_persist_mc": ("n_persist_mc", "n_pairs"),
        "frac_lens_output_sign_persist_open": ("n_persist_open", "n_pairs"),
        "frac_pairs_output_open_abs_smaller": ("n_out_open_smaller", "n_pairs"),
    }
    A["boot"] = cluster_bootstrap(per_item, specs, n_boot, seed)

    # Splits by ambiguity and polarity (descriptive, pair level).
    splits = []
    for col in ("context_condition", "question_polarity"):
        for level, g in pw.groupby(col):
            splits.append(dict(
                split=col, level=level, n_pairs=len(g), n_base_items=g["base_item"].nunique(),
                mean_mc=g["mc"].mean(), mean_open=g["open"].mean(),
                median_mc=g["mc"].median(), median_open=g["open"].median(),
                open_smaller=int(g["open_smaller"].sum()), sign_preserved=int(g["sign_preserved"].sum()),
                persist_mc=int(g["persist_mc"].sum()), persist_open=int(g["persist_open"].sum()),
                out_open_smaller=int(g["out_open_smaller"].sum()),
            ))
    A["splits"] = pd.DataFrame(splits)
    return A


def build_results(A: dict, summary_checks: dict, n_boot: int, seed: int) -> pd.DataFrame:
    R = Results()
    per_layer, pw, ev, per_item, boot = A["per_layer"], A["pw"], A["ev"], A["per_item"], A["boot"]
    row_stats = A["late_row_stats"]

    for name, info in summary_checks.items():
        R.add("validation_week3_summary", "max_abs_diff_mean_vs_recomputed", info["max_abs_diff_mean"],
              "summary-table group", subset=name, layer_range="L0-L22", denominator=info["n_groups"],
              note="Old Week 3 summary table; recomputed from raw lens rows")

    for _, r in per_layer.iterrows():
        lr = f"L{int(r['layer'])}"
        unit = "pair-level row (1 per pair at this layer)"
        R.add("per_layer_matched", "median_abs_rank_gap", r["mc_median_abs"], unit, MC, layer_range=lr,
              denominator=N_PAIRS)
        R.add("per_layer_matched", "median_abs_rank_gap", r["open_median_abs"], unit, OPEN, layer_range=lr,
              denominator=N_PAIRS)
        R.add("per_layer_matched", "pairs_open_abs_smaller", r["open_smaller_abs"] / N_PAIRS, unit,
              layer_range=lr, numerator=int(r["open_smaller_abs"]), denominator=N_PAIRS)
        R.add("per_layer_matched", "pairs_sign_agree", r["sign_agree"] / N_PAIRS, unit,
              layer_range=lr, numerator=int(r["sign_agree"]), denominator=N_PAIRS)

    for f in FORMATS:
        R.add("week3_reproduced", "row_mean_rank_gap", float(row_stats.loc[f, "mean"]), "lens row",
              f, denominator=int(row_stats.loc[f, "count"]), note="Old Week 3 finding, reproduced")
        R.add("late_window", "row_median_rank_gap", float(row_stats.loc[f, "median"]), "lens row",
              f, denominator=int(row_stats.loc[f, "count"]))
        col = "mc" if f == MC else "open"
        R.add("late_window", "pair_avg_mean_rank_gap", float(pw[col].mean()), "evaluation L15-L22 average",
              f, denominator=N_PAIRS, note="Equals row mean because every evaluation has 8 late rows")
        R.add("late_window", "pair_avg_median_rank_gap", float(pw[col].median()), "evaluation L15-L22 average",
              f, denominator=N_PAIRS)
        R.add("late_window", "pair_avg_trimmed10_mean_rank_gap", trimmed_mean(pw[col]),
              "evaluation L15-L22 average", f, denominator=N_PAIRS, note="drops 4 lowest and 4 highest")
        R.add("late_window", "evals_constant_sign_all_late_layers", A["late_const_sign"][f] / N_PAIRS,
              "evaluation", f, numerator=A["late_const_sign"][f], denominator=N_PAIRS)

    R.add("week3_reproduced", "pairs_open_abs_smaller", pw["open_smaller"].mean(), "matched pair",
          numerator=int(pw["open_smaller"].sum()), denominator=N_PAIRS, note="Old Week 3 finding, reproduced")
    R.add("week3_reproduced", "pairs_sign_preserved", pw["sign_preserved"].mean(), "matched pair",
          numerator=int(pw["sign_preserved"].sum()), denominator=N_PAIRS, note="Old Week 3 finding, reproduced")
    R.add("paired_abs_change", "median_delta_abs_open_minus_mc", float(pw["delta_abs"].median()),
          "matched pair", denominator=N_PAIRS)
    R.add("paired_abs_change", "median_log10_abs_ratio_open_over_mc", float(pw["log10_abs_ratio"].median()),
          "matched pair", denominator=N_PAIRS, note="log10((|open|+1)/(|mc|+1))")

    for f in FORMATS:
        g = ev[ev["prompt_format"] == f]
        R.add("lens_vs_output", "output_layer_mean_rank_gap", float(g["output_gap"].mean()),
              "evaluation (output layer 23)", f, layer_range="L23", denominator=len(g))
        R.add("lens_vs_output", "output_layer_median_rank_gap", float(g["output_gap"].median()),
              "evaluation (output layer 23)", f, layer_range="L23", denominator=len(g))
        R.add("lens_vs_output", "sign_persists_lens_to_output", float(g["sign_persists"].mean()),
              "evaluation", f, layer_range=f"{LATE_LABEL} vs L23",
              numerator=int(g["sign_persists"].sum()), denominator=len(g))
        R.add("lens_vs_output", "output_abs_smaller_than_lens_abs", float(g["output_abs_smaller"].mean()),
              "evaluation", f, layer_range=f"{LATE_LABEL} vs L23",
              numerator=int(g["output_abs_smaller"].sum()), denominator=len(g))
        R.add("lens_vs_output", "spearman_lens_avg_vs_output_gap", spearman(g["lens_late_avg"], g["output_gap"]),
              "evaluation", f, layer_range=f"{LATE_LABEL} vs L23", denominator=len(g),
              note="descriptive rank correlation; no significance test")
    R.add("lens_vs_output", "output_pairs_open_abs_smaller", float(pw["out_open_smaller"].mean()),
          "matched pair (output layer 23)", layer_range="L23",
          numerator=int(pw["out_open_smaller"].sum()), denominator=N_PAIRS)
    R.add("lens_vs_output", "output_pairs_sign_preserved", float(pw["out_sign_preserved"].mean()),
          "matched pair (output layer 23)", layer_range="L23",
          numerator=int(pw["out_sign_preserved"].sum()), denominator=N_PAIRS)

    top = per_item["share_abs_mc"].idxmax()
    R.add("outlier_item", "largest_base_item_share_of_sum_abs_mc", float(per_item.loc[top, "share_abs_mc"]),
          "base item", MC, subset=top, denominator=N_BASE_ITEMS)
    items_all4 = int((per_item["n_open_smaller"] == 4).sum())
    R.add("outlier_item", "base_items_open_abs_smaller_in_all_4_pairs", items_all4 / N_BASE_ITEMS,
          "base item", numerator=items_all4, denominator=N_BASE_ITEMS)
    items_mean = int((per_item["mean_open"].abs() < per_item["mean_mc"].abs()).sum())
    R.add("outlier_item", "base_items_open_abs_smaller_at_item_mean", items_mean / N_BASE_ITEMS,
          "base item", numerator=items_mean, denominator=N_BASE_ITEMS)

    method = f"base-item cluster bootstrap, 95% percentile, B={n_boot}, seed={seed}"
    for name, b in boot.items():
        R.add("uncertainty_base_item", name, b["estimate"], "matched pair, resampled by base item",
              denominator=N_PAIRS, lo=b["lo"], hi=b["hi"], method=method,
              note=f"leave-one-base-item-out range [{b['lobo_min']:.6g}, {b['lobo_max']:.6g}]")

    for _, s in A["splits"].iterrows():
        sub = f"{s['split']}={s['level']}"
        n = int(s["n_pairs"])
        nb = int(s["n_base_items"])
        R.add("split", "pair_avg_mean_rank_gap", float(s["mean_mc"]), "matched pair", MC, sub,
              denominator=n, n_base_items=nb)
        R.add("split", "pair_avg_mean_rank_gap", float(s["mean_open"]), "matched pair", OPEN, sub,
              denominator=n, n_base_items=nb)
        R.add("split", "pairs_open_abs_smaller", s["open_smaller"] / n, "matched pair", subset=sub,
              numerator=int(s["open_smaller"]), denominator=n, n_base_items=nb)
        R.add("split", "pairs_sign_preserved", s["sign_preserved"] / n, "matched pair", subset=sub,
              numerator=int(s["sign_preserved"]), denominator=n, n_base_items=nb)
    return R.frame()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_layerwise(per_layer: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    x = per_layer["layer"].to_numpy()
    for f, color, key in ((MC, "tab:red", "mc"), (OPEN, "tab:blue", "open")):
        ax1.plot(x, per_layer[f"{key}_median_abs"], marker="o", ms=3, color=color, label=f"{f} median |gap|")
        ax1.fill_between(x, per_layer[f"{key}_q25_abs"], per_layer[f"{key}_q75_abs"], color=color, alpha=0.15,
                         label=f"{f} IQR")
    ax1.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH)
    ax1.set_ylabel("|rank_gap| across 40 pairs (symlog)")
    ax1.set_title("Saved Week 3 Jacobian-lens readouts: matched MC vs open-ended |rank_gap| per layer")
    ax1.legend(fontsize=8, loc="upper left")
    ax2.plot(x, per_layer["open_smaller_abs"], marker="o", ms=3, color="tab:purple",
             label="pairs with |open| < |MC|")
    ax2.plot(x, per_layer["sign_agree"], marker="s", ms=3, color="tab:green", label="pairs with same sign")
    ax2.axhline(N_PAIRS, color="grey", lw=0.8, ls="--")
    ax2.set_ylim(0, N_PAIRS + 2)
    ax2.set_ylabel(f"count of {N_PAIRS} pairs")
    ax2.set_xlabel("lens layer")
    ax2.legend(fontsize=8, loc="lower left")
    for ax in (ax1, ax2):
        ax.axvspan(LATE_LO - 0.5, LATE_HI + 0.5, color="gold", alpha=0.12)
        ax.grid(alpha=0.3)
    ax2.set_xticks(LENS_LAYERS)
    fig.text(0.01, 0.005, "Shaded: L15-L22 window. rank_gap = counter_rank - stereotype_rank (negative: counter "
             "target ranks stronger). Pair = same item/context/polarity; 10 independent base items.",
             fontsize=7)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_lens_output_pairs(ev: pd.DataFrame, pw: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    for f, color in ((MC, "tab:red"), (OPEN, "tab:blue")):
        g = ev[ev["prompt_format"] == f]
        k = int(g["sign_persists"].sum())
        ax1.scatter(g["lens_late_avg"], g["output_gap"], s=22, alpha=0.75, color=color,
                    label=f"{f} (sign persists {k}/{len(g)})")
    ax1.set_xlabel(f"{LATE_LABEL} lens average rank_gap (symlog)")
    ax1.set_ylabel("saved output-layer (L23) rank_gap (symlog)")
    ax1.set_title("Per evaluation: internal lens gap vs output-layer token-rank gap")

    markers = {"neg": "o", "nonneg": "^"}
    colors = {"ambig": "tab:orange", "disambig": "tab:cyan"}
    for (ctx, pol), g in pw.groupby(["context_condition", "question_polarity"]):
        ax2.scatter(g["mc"], g["open"], s=28, alpha=0.8, color=colors.get(ctx, "grey"),
                    marker=markers.get(pol, "o"), label=f"{ctx} / {pol} (n={len(g)})")
    lim = float(max(pw["abs_mc"].max(), pw["abs_open"].max(), 1.0)) * 1.5
    line = np.linspace(-lim, lim, 401)
    ax2.plot(line, line, color="grey", lw=0.8, ls="--", label="|open| = |MC|")
    ax2.plot(line, -line, color="grey", lw=0.8, ls="--")
    ax2.set_xlabel(f"MC {LATE_LABEL} pair average rank_gap (symlog)")
    ax2.set_ylabel(f"open-ended {LATE_LABEL} pair average rank_gap (symlog)")
    ax2.set_title(f"Matched pairs: |open| < |MC| in {int(pw['open_smaller'].sum())}/{N_PAIRS}, "
                  f"same sign in {int(pw['sign_preserved'].sum())}/{N_PAIRS}")
    for ax in (ax1, ax2):
        ax.set_xscale("symlog", linthresh=SYMLOG_LINTHRESH)
        ax.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH)
        ax.axhline(0, color="black", lw=0.6)
        ax.axvline(0, color="black", lw=0.6)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.text(0.01, 0.005, "Token ranks only; no generated answers are inferred. Descriptive; 10 independent base "
             "items.", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def n(x, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:,.{nd}f}"


def md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def write_report(path: Path, A: dict, log: list[str], hashes: dict, summary_checks: dict,
                 n_boot: int, seed: int, out_names: dict) -> None:
    per_layer, pw, ev, per_item, boot, splits = (A["per_layer"], A["pw"], A["ev"], A["per_item"],
                                                 A["boot"], A["splits"])
    rs = A["late_row_stats"]
    early = per_layer[per_layer["layer"] < LATE_LO]
    latel = per_layer[per_layer["layer"] >= LATE_LO]

    def rng_(s):
        return f"{int(s.min())}-{int(s.max())}"

    def ci(name, nd=1):
        b = boot[name]
        return (f"{n(b['estimate'], nd)} [95% base-item bootstrap {n(b['lo'], nd)}, {n(b['hi'], nd)}; "
                f"LOBO range {n(b['lobo_min'], nd)} to {n(b['lobo_max'], nd)}]")

    L: list[str] = []
    L.append("# Offline evaluation of saved Week 3 Jacobian-lens bias readouts")
    L.append("")
    L.append("Generated by `evaluate_saved_week3.py`. Inputs are only the saved public CSVs in "
             "`Week 3/results/`. No model was loaded, downloaded or run; no GPU, network or paid compute "
             "was used; the source CSVs were not modified (SHA-256 identical before and after).")
    L.append("")
    L.append("**Sign convention.** `rank_gap = counter_rank - stereotype_rank`; negative means the "
             "counter-stereotype target text ranks stronger than the stereotype target text. Ranks are "
             "token ranks of target texts in a readout at the saved analysis position. They are not "
             "generated answers, and nothing here says what the model would have generated.")
    L.append("")
    L.append("**Units and denominators.** 10 independent base items -> 40 matched MC/open pairs "
             "(item x context x polarity) -> 80 evaluations (40 per format). The "
             f"{LATE_LABEL} window has 8 lens rows per evaluation, so 320 rows per format. Each result "
             "says which unit it uses. Pair-level counts (k/40) are not 40 independent observations; "
             "intervals resample the 10 base items.")
    L.append("")
    L.append("## Inputs and validation")
    L.append("")
    L.append(md_table(["file", "sha256 (first 16)"], [[f"`{k}`", v[:16]] for k, v in hashes.items()]))
    L.append("")
    L.append("All checks below passed before any output was written:")
    L.append("")
    L += [f"- {m}" for m in log]
    L.append("")

    L.append("## 1. Old Week 3 findings (reproduced from the saved CSVs, not new)")
    L.append("")
    L.append(f"- The three saved summary tables (lens layers 0-22; rank_gap mean/median/count by "
             f"layer x format, layer x context x format, layer x polarity x format) match a recomputation "
             f"from the raw lens rows (largest mean difference "
             f"{max(v['max_abs_diff_mean'] for v in summary_checks.values()):.3g}).")
    L.append(f"- {LATE_LABEL} row-level mean rank_gap: MC {n(rs.loc[MC, 'mean'], 6)} "
             f"(n={int(rs.loc[MC, 'count'])} rows), open-ended {n(rs.loc[OPEN, 'mean'], 6)} "
             f"(n={int(rs.loc[OPEN, 'count'])} rows).")
    L.append(f"- The open-ended |pair average| is smaller than the MC one in "
             f"{int(pw['open_smaller'].sum())}/{N_PAIRS} matched pairs, and the pair-average sign is the same "
             f"across formats in {int(pw['sign_preserved'].sum())}/{N_PAIRS} pairs.")
    L.append("")

    L.append("## 2. New offline comparisons (computed here from the same saved readouts)")
    L.append("")
    L.append(f"### 2.1 {LATE_LABEL} central tendency: rows vs pair averages")
    L.append("")
    rows = []
    for f, col in ((MC, "mc"), (OPEN, "open")):
        rows.append([f, n(rs.loc[f, "mean"]), n(rs.loc[f, "median"]), int(rs.loc[f, "count"]),
                     n(pw[col].mean()), n(pw[col].median()), n(trimmed_mean(pw[col])), N_PAIRS,
                     f"{A['late_const_sign'][f]}/{N_PAIRS}"])
    L.append(md_table(["format", "row mean", "row median", "rows", "pair-avg mean", "pair-avg median",
                       "pair-avg 10% trimmed mean", "pairs", "evals with one sign at all 8 late layers"], rows))
    L.append("")
    L.append("The row mean and the pair-average mean are the same number, because every evaluation has "
             "exactly 8 late rows. Medians and trimmed means are reported because the rank scale is "
             "heavy-tailed.")
    L.append("")
    L.append("### 2.2 Paired absolute-gap change (open-ended minus MC)")
    L.append("")
    L.append(f"- Median of |open| - |MC| over {N_PAIRS} pairs: {n(pw['delta_abs'].median())}; "
             f"median log10((|open|+1)/(|MC|+1)): {n(pw['log10_abs_ratio'].median(), 3)}.")
    L.append(f"- Mean |open| - |MC|: {ci('mean_delta_abs_open_minus_mc')}.")
    L.append(f"- Mean log10 ratio: {ci('mean_log10_abs_ratio_open_over_mc', 3)}.")
    L.append(f"- Share of pairs with |open| < |MC|: {ci('frac_pairs_open_abs_smaller', 3)}.")
    L.append("")
    L.append("### 2.3 Per-layer magnitude and sign stability (pair level, 40 pairs per layer)")
    L.append("")
    L.append(f"- Pairs with |open| < |MC|: range {rng_(early['open_smaller_abs'])}/40 over L0-L14 and "
             f"{rng_(latel['open_smaller_abs'])}/40 over {LATE_LABEL}.")
    L.append(f"- Pairs with the same sign in both formats: range {rng_(early['sign_agree'])}/40 over L0-L14 and "
             f"{rng_(latel['sign_agree'])}/40 over {LATE_LABEL}.")
    L.append(f"- See `{out_names['fig1']}` for median |rank_gap| with IQR at every layer.")
    L.append("")
    L.append(f"### 2.4 Internal lens gap vs saved output-layer (L23) gap, same evaluation and targets")
    L.append("")
    rows = []
    for f in FORMATS:
        g = ev[ev["prompt_format"] == f]
        rows.append([f, len(g), n(g["lens_late_avg"].median()), n(g["output_gap"].median()),
                     f"{int(g['sign_persists'].sum())}/{len(g)}", f"{int(g['output_abs_smaller'].sum())}/{len(g)}",
                     n(spearman(g["lens_late_avg"], g["output_gap"]), 2)])
    L.append(md_table(["format", "evals", f"median {LATE_LABEL} lens gap", "median L23 output gap",
                       "sign persists", "|output| < |lens|", "Spearman rho (descriptive)"], rows))
    L.append("")
    L.append(f"- At the output layer, |open| < |MC| in {int(pw['out_open_smaller'].sum())}/{N_PAIRS} pairs and "
             f"the sign matches across formats in {int(pw['out_sign_preserved'].sum())}/{N_PAIRS} pairs.")
    L.append(f"- Sign persistence with base-item intervals: MC {ci('frac_lens_output_sign_persist_mc', 3)}; "
             f"open-ended {ci('frac_lens_output_sign_persist_open', 3)}.")
    L.append("- This only compares token-rank gaps between two saved readouts. It does not show what the model "
             "generated, and it does not show that the lens gap causes the output gap.")
    L.append("")
    L.append("### 2.5 Outlier sensitivity and base-item consistency")
    L.append("")
    top = per_item["share_abs_mc"].idxmax()
    L.append(f"- Base item `{top}` contributes {n(100 * per_item.loc[top, 'share_abs_mc'])}% of the summed "
             f"|MC pair average| (10 items).")
    L.append(f"- MC mean: {ci('mean_pair_avg_gap_mc')}.")
    L.append(f"- Open-ended mean: {ci('mean_pair_avg_gap_open')}.")
    L.append(f"- Base items with |open| < |MC| in all 4 of their pairs: "
             f"{int((per_item['n_open_smaller'] == 4).sum())}/10; at the base-item mean: "
             f"{int((per_item['mean_open'].abs() < per_item['mean_mc'].abs()).sum())}/10.")
    L.append("")
    rows = [[f"`{idx}`", f"{int(r['n_open_smaller'])}/4", f"{int(r['n_sign_preserved'])}/4",
             n(r["mean_mc"]), n(r["mean_open"]), f"{n(100 * r['share_abs_mc'])}%"]
            for idx, r in per_item.sort_values("share_abs_mc", ascending=False).iterrows()]
    L.append(md_table(["base item", "|open|<|MC| pairs", "same sign pairs", "mean MC gap", "mean open gap",
                       "share of sum |MC|"], rows))
    L.append("")
    L.append("### 2.6 Uncertainty")
    L.append("")
    L.append(f"Intervals come from a deterministic cluster bootstrap that resamples the 10 base items with "
             f"replacement (B={n_boot}, seed={seed}, 95% percentile), plus leave-one-base-item-out (LOBO) "
             f"ranges. With 10 clusters these intervals are rough and descriptive. They are not hypothesis "
             f"tests, and no p-values or significance claims are made. A proportion that is 40/40 gets a "
             f"degenerate interval, and that does not mean it is certain.")
    L.append("")
    L.append("### 2.7 Ambiguity and polarity splits (descriptive, pair level)")
    L.append("")
    rows = [[f"{s['split']}={s['level']}", int(s["n_pairs"]), int(s["n_base_items"]), n(s["mean_mc"]),
             n(s["mean_open"]), n(s["median_mc"]), n(s["median_open"]),
             f"{int(s['open_smaller'])}/{int(s['n_pairs'])}", f"{int(s['sign_preserved'])}/{int(s['n_pairs'])}",
             f"{int(s['persist_mc'])}/{int(s['n_pairs'])}", f"{int(s['persist_open'])}/{int(s['n_pairs'])}"]
            for _, s in splits.iterrows()]
    L.append(md_table(["subset", "pairs", "base items", "MC mean", "open mean", "MC median", "open median",
                       "|open|<|MC|", "same sign", "lens->L23 sign MC", "lens->L23 sign open"], rows))
    L.append("")
    L.append("Each split has 20 pairs from the same 10 base items, so the splits are not independent of "
             "each other. No split difference is tested.")
    L.append("")

    L.append("## 3. Causal RQ4: still unanswered")
    L.append("")
    L.append("Nothing here answers the causal research question (RQ4). No activation patching, ablation, "
             "steering or other intervention was run, and none can be run without model inference, which "
             "this offline evaluation leaves out on purpose. Every result above is a descriptive or "
             "correlational summary of saved token-rank readouts. Agreement between lens and output-layer "
             "gaps does not show that internal representations cause output rankings or answer choices.")
    L.append("")
    L.append("## Limitations")
    L.append("")
    L.append("- There are only 10 base items from one saved Week 3 run of one model. Pair- and "
             "evaluation-level counts are clustered within base items.")
    L.append("- Ranks are heavy-tailed and bounded by vocabulary size, so means are dominated by a few "
             "very large MC gaps (see 2.5). The medians, trimmed means and log ratios are there for that "
             "reason.")
    L.append("- The ranks belong to target-text tokens at one saved analysis position. The way multi-token "
             "targets were combined into one rank comes from Week 3 and is not re-checked here. MC prompts may "
             "favour answer-letter tokens over answer text, which could inflate MC rank magnitudes. That is "
             "a hypothesis, not something tested here.")
    L.append("- The output-layer row is a token-rank readout, not a generated answer or an accuracy measure.")
    L.append(f"- Summary-table agreement is checked to an absolute tolerance of {SUMMARY_ATOL} to allow for "
             "CSV rounding.")
    L.append(f"- Zero gaps count as their own sign (np.sign = 0) in all sign comparisons.")
    L.append("")
    L.append(f"Outputs: `{out_names['csv']}`, `{out_names['fig1']}`, `{out_names['fig2']}`, this report.")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo-root", default=None,
                   help="Repository root (default: derived from this script's location).")
    p.add_argument("--n-boot", type=int, default=10000, help="Base-item bootstrap resamples (default 10000).")
    p.add_argument("--seed", type=int, default=20261020, help="Bootstrap seed (default 20261020).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else DEFAULT_REPO_ROOT
    results_dir = repo_root / "Week 3" / "results"
    paths = {name: results_dir / name for name in [RAW_NAME, *SUMMARY_SPECS]}
    try:
        for p in paths.values():
            if not p.is_file():
                raise ValidationError(f"FAILED: missing input {p}")
        hashes = {name: sha256(p) for name, p in paths.items()}

        log: list[str] = []
        raw = load_raw(paths[RAW_NAME])
        validate_raw(raw, log)
        lens = raw[raw["readout"] == LENS].copy()
        out = raw[raw["readout"] == OUTPUT].copy()
        summary_checks = {name: validate_summary(paths[name], gcols, lens, log)
                          for name, gcols in SUMMARY_SPECS.items()}
        A = analyse(lens, out, log, args.n_boot, args.seed)
        results = build_results(A, summary_checks, args.n_boot, args.seed)
        if {name: sha256(p) for name, p in paths.items()} != hashes:
            raise ValidationError("FAILED: source CSVs changed during the run")
    except ValidationError as exc:
        print(exc, file=sys.stderr)
        print("No outputs were written.", file=sys.stderr)
        return 1

    out_names = {"csv": "offline_evaluation_results.csv", "fig1": "fig_layerwise_format_gaps.png",
                 "fig2": "fig_lens_vs_output_and_pairs.png", "report": "offline_evaluation_report.md"}
    results.to_csv(SCRIPT_DIR / out_names["csv"], index=False)
    plot_layerwise(A["per_layer"], SCRIPT_DIR / out_names["fig1"])
    plot_lens_output_pairs(A["ev"], A["pw"], SCRIPT_DIR / out_names["fig2"])
    write_report(SCRIPT_DIR / out_names["report"], A, log, hashes, summary_checks,
                 args.n_boot, args.seed, out_names)

    if {name: sha256(p) for name, p in paths.items()} != hashes:
        print("FAILED: source CSVs changed during the run", file=sys.stderr)
        return 1
    print(f"Validation passed ({len(log)} checks). Wrote to {SCRIPT_DIR}:")
    for v in out_names.values():
        print(f"  {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
