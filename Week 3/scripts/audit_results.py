#!/usr/bin/env python3
"""Deterministic Week 3 result-integrity audit (read-only).

Never imports torch/transformers/jlens and never loads a model. Inspects
only already-written real artifacts (week3_raw_results.csv,
week3_recovery_state.json, week3_memory_log.jsonl) produced by
run_week3_experiment.py's authorized inference mode.

Exit code 0 = PASS, non-zero = FAIL (prints each issue found).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

WEEK3_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = WEEK3_DIR / "results"
RAW_CSV = RESULTS_DIR / "week3_raw_results.csv"
RECOVERY_STATE = RESULTS_DIR / "week3_recovery_state.json"
MEMORY_LOG = RESULTS_DIR / "week3_memory_log.jsonl"

EXPECTED_MODEL = "Qwen/Qwen3.5-0.8B"
EXPECTED_LENS_REPO = "neuronpedia/jacobian-lens"
EXPECTED_LENS_REVISION = "qwen-n1000"
EXPECTED_LENS_FILE = "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"
CONDITIONS_PER_BASE_ITEM = 4
FORMATS = {"multiple_choice", "open_ended"}
CONTEXT_CONDITIONS = {"ambig", "disambig"}
POLARITIES = {"neg", "nonneg"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expected-base-items", type=int, required=True)
    args = ap.parse_args()
    issues: list[str] = []

    if not RAW_CSV.exists():
        print(f"RESULT_AUDIT: FAIL\n  - {RAW_CSV} does not exist")
        sys.exit(1)

    with open(RAW_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("RESULT_AUDIT: FAIL\n  - raw results CSV is empty")
        sys.exit(1)

    expected_evals = args.expected_base_items * CONDITIONS_PER_BASE_ITEM * len(FORMATS)
    base_items = sorted({r["base_item_id"] for r in rows})
    if len(base_items) != args.expected_base_items:
        issues.append(f"expected {args.expected_base_items} base items, found {len(base_items)}: {base_items}")
    if len(rows) < expected_evals:
        issues.append(f"expected >= {expected_evals} rows, found {len(rows)}")

    # Each prompt evaluation legitimately yields many rows (one per layer/readout), so
    # duplicates are judged at (evaluation, layer, readout) granularity.
    keys = [
        (r["base_item_id"], r["context_condition"], r["question_polarity"], r["prompt_format"], r["layer"], r["readout"])
        for r in rows
    ]
    if len(keys) != len(set(keys)):
        issues.append("duplicate (base_item_id, context_condition, question_polarity, prompt_format, layer, readout) rows found")
    n_evals = len({k[:4] for k in keys})
    if n_evals != expected_evals:
        issues.append(f"expected {expected_evals} distinct prompt evaluations, found {n_evals}")

    by_key: dict[tuple, set[str]] = {}
    for r in rows:
        k = (r["base_item_id"], r["context_condition"], r["question_polarity"])
        by_key.setdefault(k, set()).add(r["prompt_format"])
    incomplete = [k for k, v in by_key.items() if v != FORMATS]
    if incomplete:
        issues.append(f"{len(incomplete)} base/condition/polarity combo(s) missing a MC/open-ended pair member: {incomplete[:5]}")

    conds = {r["context_condition"] for r in rows}
    if not CONTEXT_CONDITIONS <= conds:
        issues.append(f"context_condition coverage incomplete, expected {CONTEXT_CONDITIONS}, found {sorted(conds)}")

    pols = {r["question_polarity"] for r in rows}
    if not POLARITIES <= pols:
        issues.append(f"question_polarity coverage incomplete, expected {POLARITIES}, found {sorted(pols)}")

    bad_source = [
        r for r in rows if r["prompt_format"] == "open_ended" and r.get("source_token", "").strip() != ":"
    ]
    if bad_source:
        issues.append(
            f"{len(bad_source)} open_ended row(s) have source_token != ':' "
            "(analysis position is not the final 'Answer:' colon)"
        )

    bad_rankgap = 0
    for r in rows:
        try:
            float(r["rank_gap"])
        except (ValueError, TypeError, KeyError):
            bad_rankgap += 1
    if bad_rankgap:
        issues.append(f"{bad_rankgap} row(s) have non-numeric/missing rank_gap")

    empty_prompt = [r for r in rows if not r.get("prompt", "").strip()]
    if empty_prompt:
        issues.append(f"{len(empty_prompt)} row(s) have an empty prompt field")

    if not RECOVERY_STATE.exists():
        issues.append(f"{RECOVERY_STATE} missing")
    else:
        state = json.loads(RECOVERY_STATE.read_text())
        for field, expected in [
            ("model", EXPECTED_MODEL),
            ("lens_repo", EXPECTED_LENS_REPO),
            ("lens_revision", EXPECTED_LENS_REVISION),
            ("lens_file", EXPECTED_LENS_FILE),
        ]:
            if state.get(field) != expected:
                issues.append(f"recovery state {field}={state.get(field)!r}, expected {expected!r}")
        if state.get("needs_finalization"):
            issues.append(
                f"recovery state needs_finalization=true (stopped_reason={state.get('stopped_reason')!r}); "
                "run did not complete cleanly"
            )

    if not MEMORY_LOG.exists():
        issues.append(f"{MEMORY_LOG} missing")

    if issues:
        print("RESULT_AUDIT: FAIL")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)

    print(f"RESULT_AUDIT: PASS ({len(rows)} rows, {len(base_items)} base items)")
    sys.exit(0)


if __name__ == "__main__":
    main()
