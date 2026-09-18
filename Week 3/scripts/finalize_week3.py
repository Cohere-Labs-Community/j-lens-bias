#!/usr/bin/env python3
"""Cheap Week 3 finalizer: reads Week 3/results/week3_raw_results.csv (already
produced by run_week3_experiment.py's authorized inference mode) and builds
summaries/figures + refreshes the submission notebook's result cells.

Never loads torch/transformers/jlens and never reloads the model or lens --
only pandas/matplotlib/nbformat/nbclient over the already-saved CSV, mirroring
Week 2/scripts/finalize_week2.py's architecture.

NOTE (this authoring session): this script is NOT executed here. Per the
static-development-only constraint for this session, it is validated only via
py_compile/AST parsing. It becomes runnable once Week 3 inference has actually
produced week3_raw_results.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import pandas as pd
from nbclient import NotebookClient

WEEK3_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = WEEK3_DIR / "results"
SUBMISSIONS_DIR = WEEK3_DIR / "submissions"
RAW_CSV = RESULTS_DIR / "week3_raw_results.csv"
NOTEBOOK_PATH = SUBMISSIONS_DIR / "MelodyResearch_week_3.ipynb"


def build_summaries_and_figures() -> tuple[dict, "pd.DataFrame"] | None:
    if not RAW_CSV.exists():
        print(
            f"NOT AVAILABLE FROM CURRENT FILES: {RAW_CSV} does not exist yet -- "
            f"Week 3 inference has not been run. Nothing to finalize."
        )
        return None

    df = pd.read_csv(RAW_CSV)
    if df.empty:
        print(f"{RAW_CSV} exists but is empty. Nothing to finalize.")
        return None

    lens_df = df[df.readout == "jacobian_lens"].copy()

    # Summary 1: rank_gap by layer x prompt_format (the core MC-vs-open-ended comparison)
    by_format = (
        lens_df.groupby(["layer", "prompt_format"])["rank_gap"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    by_format.to_csv(RESULTS_DIR / "week3_summary_by_layer_format.csv", index=False)

    # Summary 2: rank_gap by layer x context_condition x prompt_format
    by_cond_format = (
        lens_df.groupby(["layer", "context_condition", "prompt_format"])["rank_gap"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    by_cond_format.to_csv(RESULTS_DIR / "week3_summary_by_layer_condition_format.csv", index=False)

    # Summary 3: rank_gap by layer x question_polarity x prompt_format
    by_pol_format = (
        lens_df.groupby(["layer", "question_polarity", "prompt_format"])["rank_gap"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    by_pol_format.to_csv(RESULTS_DIR / "week3_summary_by_layer_polarity_format.csv", index=False)

    # Fig 1: rank_gap by layer, multiple_choice vs open_ended (the headline comparison)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fmt, sub in by_format.groupby("prompt_format"):
        ax.plot(sub["layer"], sub["mean"], marker="o", label=f"{fmt} (mean)")
    ax.axhline(0, color="grey", linewidth=1, linestyle="--")
    ax.set_xlabel("fitted lens layer")
    ax.set_ylabel("rank_gap = counter_rank - stereotype_rank")
    ax.set_title("rank_gap by layer: multiple_choice vs. open_ended (Age, matched pairs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig1_rankgap_by_format.png", dpi=130)
    plt.close(fig)

    # Fig 2: rank_gap by layer, ambiguous vs disambiguated, split by format
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, fmt in zip(axes, ["multiple_choice", "open_ended"]):
        sub_fmt = by_cond_format[by_cond_format.prompt_format == fmt]
        for cond, sub in sub_fmt.groupby("context_condition"):
            ax.plot(sub["layer"], sub["mean"], marker="o", label=f"{cond}")
        ax.axhline(0, color="grey", linewidth=1, linestyle="--")
        ax.set_title(fmt)
        ax.set_xlabel("fitted lens layer")
        ax.legend()
    axes[0].set_ylabel("rank_gap = counter_rank - stereotype_rank")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig2_rankgap_by_condition_format.png", dpi=130)
    plt.close(fig)

    # Fig 3: rank_gap by layer, negative vs non-negative, split by format
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, fmt in zip(axes, ["multiple_choice", "open_ended"]):
        sub_fmt = by_pol_format[by_pol_format.prompt_format == fmt]
        for pol, sub in sub_fmt.groupby("question_polarity"):
            ax.plot(sub["layer"], sub["mean"], marker="o", label=f"{pol}")
        ax.axhline(0, color="grey", linewidth=1, linestyle="--")
        ax.set_title(fmt)
        ax.set_xlabel("fitted lens layer")
        ax.legend()
    axes[0].set_ylabel("rank_gap = counter_rank - stereotype_rank")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig3_rankgap_by_polarity_format.png", dpi=130)
    plt.close(fig)

    stats = {
        "n_rows": len(df),
        "n_base_items": int(df["base_item_id"].nunique()),
        "n_pairs": int(df["pair_id"].nunique()),
        "n_layers": int(lens_df["layer"].nunique()),
        "formats_present": sorted(df["prompt_format"].unique().tolist()),
    }
    return stats, df


def _find_markdown_cell(nb, prefix: str):
    for c in nb.cells:
        if c.cell_type == "markdown" and c.source.strip().startswith(prefix):
            return c
    return None


def refresh_notebook(stats: dict, df: "pd.DataFrame") -> None:
    """Mechanically refresh only the result-dependent notebook cells from
    already-computed, real Week 3 data -- no interpretation is authored here,
    only real numbers pulled directly from week3_raw_results.csv. Then
    executes the notebook in place (nbclient) so code cells (which already
    load CSV/PNG files that now exist) render real outputs. Never imports
    torch/transformers/jlens; the notebook's own code cells don't either.
    """
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    lens_df = df[df.readout == "jacobian_lens"].copy()
    by_format = lens_df.groupby("prompt_format")["rank_gap"].agg(["mean", "median", "count"])

    results_cell = _find_markdown_cell(nb, "## 19. Results")
    if results_cell is not None:
        lines = [
            "## 19. Results",
            "",
            f"Exploratory (N={stats['n_rows']} rows, {stats['n_base_items']} matched base item(s), "
            f"{stats['n_pairs']} matched MC/open-ended pair(s); no significance testing performed):",
            "",
        ]
        for fmt in stats["formats_present"]:
            if fmt in by_format.index:
                row = by_format.loc[fmt]
                lines.append(
                    f"- `{fmt}`: mean rank_gap = {row['mean']:.3f} (median {row['median']:.3f}, n={int(row['count'])})"
                )
        lines += [
            "",
            "rank_gap = counter_rank - stereotype_rank (higher = stereotype token ranked "
            "higher / more probable than the counter-stereotype token; see Section 12).",
            "See Sections 14-17 above for the full layer-wise and matched-pair breakdowns.",
        ]
        results_cell.source = "\n".join(lines)

    ex_cell = _find_markdown_cell(nb, "## 18. Representative Examples")
    if ex_cell is not None:
        lines = ["## 18. Representative Examples", ""]
        if len(lens_df):
            min_layer = lens_df["layer"].min()
            layer_rows = lens_df[lens_df.layer == min_layer]
            first_pair = layer_rows["pair_id"].iloc[0]
            pair_rows = layer_rows[layer_rows.pair_id == first_pair]
            lines.append(
                f"Real example pulled from `week3_raw_results.csv` (pair_id=`{first_pair}`, layer={int(min_layer)}):"
            )
            lines.append("")
            for _, r in pair_rows.iterrows():
                lines.append(
                    f"- `{r['prompt_format']}` / {r['context_condition']}/{r['question_polarity']}: "
                    f"stereotype_rank={r['stereotype_rank']}, counter_rank={r['counter_rank']}, "
                    f"rank_gap={r['rank_gap']}"
                )
        else:
            lines.append("No jacobian_lens rows available.")
        ex_cell.source = "\n".join(lines)

    lim_cell = _find_markdown_cell(nb, "## 20. Limitations")
    if lim_cell is not None:
        kept = [line for line in lim_cell.source.splitlines() if "No results yet" not in line]
        lim_cell.source = "\n".join(kept)

    nbformat.write(nb, NOTEBOOK_PATH)
    client = NotebookClient(
        nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(SUBMISSIONS_DIR)}}
    )
    client.execute()
    nbformat.write(nb, NOTEBOOK_PATH)
    nbformat.validate(nb)


def main() -> None:
    result = build_summaries_and_figures()
    if result is None:
        print("Finalize skipped: no Week 3 results yet.")
        return
    stats, df = result
    print(f"Finalize summary: {json.dumps(stats, indent=2)}")
    refresh_notebook(stats, df)
    print(f"Notebook refreshed and executed in place: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
