#!/usr/bin/env python3
"""Deterministic Week 3 submission/notebook audit (read-only).

Never runs inference, never invents scientific interpretation. Validates
that the final notebook and result artifacts are internally consistent and
that nothing outside Week 3 was modified by the overnight workflow.

Exit code 0 = PASS, non-zero = FAIL (prints each issue found).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WEEK3_DIR = REPO_ROOT / "Week 3"
RESULTS_DIR = WEEK3_DIR / "results"
NOTEBOOK_PATH = WEEK3_DIR / "submissions" / "MelodyResearch_week_3.ipynb"
RAW_CSV = RESULTS_DIR / "week3_raw_results.csv"

FORBIDDEN_MODEL_SUBSTRINGS = [
    "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-27B", "Qwen2B", "Qwen 2B", "Qwen-2B",
    "Qwen 4B", "Qwen-4B", "Qwen 27B", "Qwen-27B",
]
PLACEHOLDER_STRINGS = [
    "Results will be populated after the experiment is executed.",
    "No results yet.",
]
EXPECTED_FIGURES = [
    "fig1_rankgap_by_format.png",
    "fig2_rankgap_by_condition_format.png",
    "fig3_rankgap_by_polarity_format.png",
]


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    ).stdout


def main() -> None:
    issues: list[str] = []

    if not NOTEBOOK_PATH.exists():
        print(f"NOTEBOOK_AUDIT: FAIL\n  - notebook missing: {NOTEBOOK_PATH}")
        sys.exit(1)

    try:
        nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
        nbformat.validate(nb)
    except Exception as exc:  # noqa: BLE001
        print(f"NOTEBOOK_AUDIT: FAIL\n  - notebook failed to parse/validate: {exc!r}")
        sys.exit(1)

    full_text = "\n".join(
        "".join(c.get("source", "")) if isinstance(c.get("source"), list) else c.get("source", "")
        for c in nb.cells
    )
    # include rendered outputs too (numbers/text baked in by execution)
    for c in nb.cells:
        for out in c.get("outputs", []) or []:
            for v in (out.get("data", {}) or {}).values():
                if isinstance(v, str):
                    full_text += "\n" + v
                elif isinstance(v, list):
                    full_text += "\n" + "".join(v)
            if "text" in out:
                t = out["text"]
                full_text += "\n" + ("".join(t) if isinstance(t, list) else t)

    for bad in FORBIDDEN_MODEL_SUBSTRINGS:
        if bad in full_text:
            issues.append(f"forbidden model leakage found in notebook: {bad!r}")
    if "cuda" in full_text.lower():
        issues.append("forbidden 'cuda' reference found in notebook (device must be MPS)")

    for ph in PLACEHOLDER_STRINGS:
        if ph in full_text:
            issues.append(f"obsolete placeholder still present in notebook: {ph!r}")

    if not RAW_CSV.exists():
        issues.append(f"{RAW_CSV} missing -- cannot cross-check notebook numbers against real results")
    else:
        import csv as _csv

        with open(RAW_CSV, newline="") as f:
            n_rows = sum(1 for _ in _csv.DictReader(f))
        if str(n_rows) not in full_text:
            issues.append(
                f"notebook does not appear to reference the real row count ({n_rows}) from {RAW_CSV.name}"
            )

    for fig in EXPECTED_FIGURES:
        if not (RESULTS_DIR / fig).exists():
            issues.append(f"expected figure missing: {fig}")

    for exec_count_cell in [
        c for c in nb.cells if c.cell_type == "code" and "raw_csv" in "".join(c.get("source", ""))
    ]:
        if exec_count_cell.get("execution_count") is None:
            issues.append("notebook result-loading code cell was never executed (no execution_count)")

    week2_status_lines = [
        ln for ln in git("status", "--porcelain", "--", "Week 2").strip().splitlines()
        if ".DS_Store" not in ln
    ]
    if week2_status_lines:
        issues.append(f"Week 2 has uncommitted modifications (must be untouched): {week2_status_lines[:5]}")

    stage4r_status = git("status", "--porcelain").strip()
    stage4r_lines = [ln for ln in stage4r_status.splitlines() if "stage4r" in ln.lower()]
    if stage4r_lines:
        issues.append(f"stage4r-related paths appear in git status (must be untouched): {stage4r_lines}")

    if issues:
        print("NOTEBOOK_AUDIT: FAIL")
        for i in issues:
            print(f"  - {i}")
        sys.exit(1)

    print("NOTEBOOK_AUDIT: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
