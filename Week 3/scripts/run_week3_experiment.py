#!/usr/bin/env python3
"""Week 3 Jacobian Lens BBQ bias experiment runner (Cohere Labs J-Lens Bias).

Option 1 (per upstream Week 3/README.md): "Drop the multiple-choice" -- test
whether the Week 2 J-Lens bias-related readout on BBQ Age items persists when
the multiple-choice answer structure is removed.

For each of the SAME 10 matched Week 2 BBQ Age base items x 4 conditions
(ambig/disambig x neg/nonneg), this script constructs a MATCHED PAIR of
prompts differing only in whether the answer choices are present:

  multiple_choice: "{context}\\n{question}\\n(a) {ans0}\\n(b) {ans1}\\n
                     (c) {ans2}\\nAnswer:"   (verbatim Week 2 PROMPT_TEMPLATE)
  open_ended:       "{context}\\n{question}\\nAnswer:"

Context/question text, BBQ metadata, and stereotype/counter role mapping are
reused VERBATIM from Week 2's validated selection
(Week 2/data/age_selected_items.jsonl) -- this script never re-derives or
edits stereotype/counter roles and never invents new base items.

Architecture (checkpoint/resume, memory safety, PID lock, sequential
single-instance MPS inference, atomic writes) is a direct port of
Week 2/scripts/run_week2_experiment.py, with a SAFER DEFAULT specific to
Week 3 (see SAFETY MODES below) because another, unrelated experiment may
already be running on this machine when Week 3 starts.

======================== SAFETY MODES (READ FIRST) ========================
Running this script with NO flags (or --prepare-only) NEVER imports torch,
transformers, or jlens, and NEVER loads a model. It only reads the existing
Week 2 selected-items JSONL, builds the matched MC/open-ended prompt pairs,
writes them to Week 3/results/week3_planned_prompts.jsonl, prints the exact
planned model-evaluation count, and exits.

Actual inference requires BOTH of:
  --run-inference
  --i-confirm-no-heavy-experiment
Without BOTH flags present, the script exits before any model/runtime import.

--sanity-only additionally restricts an authorized inference run to exactly
ONE matched base item (its 4 BBQ conditions x 2 prompt formats = 8 forward
passes), never more, and is never silently expanded.
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
SUBMISSION_REPO = HOME / "Desktop/JLens-project/j-lens-bias"
WEEK2_DIR = SUBMISSION_REPO / "Week 2"
WEEK3_DIR = SUBMISSION_REPO / "Week 3"
WEEK2_SELECTED_ITEMS_JSONL = WEEK2_DIR / "data" / "age_selected_items.jsonl"

RESULTS_DIR = WEEK3_DIR / "results"
CKPT_DIR = RESULTS_DIR / "checkpoints"
COMPLETED_JSON = RESULTS_DIR / "week3_completed_prompts.json"
RAW_CSV = RESULTS_DIR / "week3_raw_results.csv"
MEMORY_LOG = RESULTS_DIR / "week3_memory_log.jsonl"
PID_FILE = RESULTS_DIR / "week3_runner.pid"
RECOVERY_STATE = RESULTS_DIR / "week3_recovery_state.json"
PLANNED_PROMPTS = RESULTS_DIR / "week3_planned_prompts.jsonl"

# --------------------------------------------------------------------------- #
# Model / lens config -- VERIFIED identical to Week 2
# (Week 2/scripts/run_week2_experiment.py lines 70-73 / week2_recovery_state.json)
# --------------------------------------------------------------------------- #
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"

TOP_K = 20  # VERIFIED identical to Week 2 (TOP_K = 20)
CONDITIONS_PER_BASE_ITEM = 4  # ambig/disambig x neg/nonneg (Week 2 design)
FORMATS = ["multiple_choice", "open_ended"]
HARD_CAP_BASE_ITEMS = 10  # VERIFIED identical to Week 2 (HARD_CAP_BASE_ITEMS)
STAGE_BOUNDARIES = [1, 5, 8, 10]  # base-item counts (inherited from Week 2)

PROMPT_TEMPLATE_MC = "{context}\n{question}\n(a) {ans0}\n(b) {ans1}\n(c) {ans2}\nAnswer:"
PROMPT_TEMPLATE_OPEN = "{context}\n{question}\nAnswer:"

CSV_FIELDS = [
    "category",
    "base_item_id",
    "example_id",
    "pair_id",
    "context_condition",
    "question_polarity",
    "prompt_format",
    "prompt",
    "question",
    "layer",
    "readout",
    "seq_len",
    "analysis_position",
    "source_token",
    "top20_tokens",
    "top20_ranks",
    "stereotype_group",
    "counter_group",
    "stereotype_target_text",
    "counter_target_text",
    "stereotype_constituent_tokens",
    "stereotype_constituent_ranks",
    "counter_constituent_tokens",
    "counter_constituent_ranks",
    "stereotype_rank",
    "counter_rank",
    "rank_gap",
    "unknown_answer_key",
    "correct_answer_label",
    "stereotype_answer_key",
    "counter_answer_key",
]

# ---- Memory safety thresholds ----
# FREE_PCT_FLOOR / SWAP_DRIFT_CEIL_MB / MPS_DRIFT_CEIL_MB values are VERIFIED
# identical to Week 2 (Week 2/scripts/run_week2_experiment.py lines 111-113).
FREE_PCT_FLOOR = 20
SWAP_DRIFT_CEIL_MB = 3072.0
MPS_DRIFT_CEIL_MB = 2048.0

# DESIGN_DECISION (Week 3 only, not inherited from Week 2): an unrelated
# experiment may already have consumed memory BEFORE this script starts. macOS
# swap is sticky -- ~5 GB of *historical* swap can remain after that experiment
# has exited, so an absolute swap ceiling cannot tell stale swap from NEW
# pressure caused by Week 3. The gate therefore judges: pre-load free memory
# (3 stable readings), memory-pressure level, and swap GROWTH relative to the
# BASELINE_SWAP_MB recorded immediately before model load.
LEGACY_PRELOAD_SWAP_ABS_CEIL_MB = 4096.0  # provenance only; no longer a gate
PRELOAD_FREE_PCT_FLOOR = 30  # in-run floor (was 25); stops Week 3 below this
SWAP_DRIFT_CEIL_MB = 1500.0  # stop at >= +1500MB swap vs BASELINE_SWAP_MB (was 3072)
PRELOAD_FREE_PCT_MIN = 60  # required on ALL pre-load readings before model load
PRELOAD_READINGS = 3
PRELOAD_READING_GAP_SEC = 25
PRESSURE_LEVEL_CRITICAL = 4  # kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical
BASELINE_SWAP_FILE = RESULTS_DIR / "week3_baseline_swap_mb.txt"


# --------------------------------------------------------------------------- #
# Run lock (ported verbatim from Week 2's acquire_lock/release_lock/
# check_no_other_qwen_process, renamed for the Week 3 script/PID file)
# --------------------------------------------------------------------------- #


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if PID_FILE.exists():
        try:
            existing_pid = int(PID_FILE.read_text().strip())
        except ValueError:
            existing_pid = -1
        if existing_pid > 0 and _pid_is_alive(existing_pid):
            print(
                f"REFUSING TO START: another Week 3 runner appears active "
                f"(pid {existing_pid} in {PID_FILE}). Not launching a second "
                f"model process."
            )
            sys.exit(3)
        print(f"Stale lock file (pid {existing_pid} not alive); replacing.")
    PID_FILE.write_text(str(os.getpid()))


def release_lock() -> None:
    try:
        if PID_FILE.exists() and PID_FILE.read_text().strip() == str(os.getpid()):
            PID_FILE.unlink()
    except OSError:
        pass


def check_no_other_week3_process() -> None:
    """Refuse only on a genuine duplicate of THIS script (mirrors Week 2's
    check_no_other_qwen_process). Per task constraints this script must never
    try to identify or act on the OTHER, unrelated experiment already running
    on this machine -- it only prevents launching a second copy of itself."""
    try:
        out = subprocess.run(["ps", "-eo", "pid,command"], capture_output=True, text=True).stdout
    except Exception:
        return
    my_pid = os.getpid()
    my_ppid = os.getppid()
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        pid_str, command = parts
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if pid in (my_pid, my_ppid):
            continue
        first_token = command.split()[0] if command.split() else ""
        is_python = Path(first_token).name.startswith("python")
        if is_python and "run_week3_experiment.py" in command.lower() and _pid_is_alive(pid):
            print(f"REFUSING TO START: found another Week 3 runner process (pid {pid}): {command}")
            sys.exit(3)


# --------------------------------------------------------------------------- #
# Memory safety
# --------------------------------------------------------------------------- #


def check_memory(torch_mod=None) -> dict:
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        out = ""
    m = re.search(r"free percentage:\s*(\d+)%", out)
    free_pct = int(m.group(1)) if m else None

    try:
        swap_out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        swap_out = ""
    m2 = re.search(r"used\s*=\s*([\d.]+)M", swap_out)
    swap_used_mb = float(m2.group(1)) if m2 else None

    try:
        pl_out = subprocess.run(
            ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        pressure_level = int(pl_out) if pl_out else None
    except Exception:
        pressure_level = None

    mps_alloc_mb = mps_driver_mb = None
    if torch_mod is not None and torch_mod.backends.mps.is_available():
        try:
            mps_alloc_mb = torch_mod.mps.current_allocated_memory() / 1e6
            mps_driver_mb = torch_mod.mps.driver_allocated_memory() / 1e6
        except Exception:
            pass

    return {
        "ts": time.time(),
        "free_pct": free_pct,
        "swap_used_mb": swap_used_mb,
        "pressure_level": pressure_level,
        "mps_alloc_mb": mps_alloc_mb,
        "mps_driver_mb": mps_driver_mb,
    }


def log_memory(tag: str, snap: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_LOG, "a") as f:
        f.write(json.dumps({"tag": tag, **snap}) + "\n")
    print(
        f"[memory:{tag}] free={snap['free_pct']}% swap={snap['swap_used_mb']}MB pressure_level={snap.get('pressure_level')} "
        f"mps_alloc={snap['mps_alloc_mb']}MB mps_driver={snap['mps_driver_mb']}MB",
        file=sys.stderr,
    )


def memory_is_safe(snap: dict, baseline: dict) -> tuple[bool, str]:
    """Inherited Week 2 drift-based gates PLUS Week-3-only absolute pre-load
    gates. Unlike Week 2, an unreadable memory/swap reading is treated as
    UNSAFE (fail-closed) rather than silently skipped, because Week 3 must
    assume another experiment may already be consuming memory."""
    if snap["free_pct"] is None:
        return False, "free_pct could not be read (memory_pressure unavailable) -- fail-closed"
    if snap["free_pct"] < FREE_PCT_FLOOR:
        return False, f"free_pct {snap['free_pct']}% < floor {FREE_PCT_FLOOR}%"
    if snap["free_pct"] < PRELOAD_FREE_PCT_FLOOR:
        return False, f"free_pct {snap['free_pct']}% < Week-3 stricter floor {PRELOAD_FREE_PCT_FLOOR}%"
    if snap["swap_used_mb"] is None:
        return False, "swap_used_mb could not be read (sysctl unavailable) -- fail-closed"
    if snap.get("pressure_level") is not None and snap["pressure_level"] >= PRESSURE_LEVEL_CRITICAL:
        return False, f"memory pressure level {snap['pressure_level']} (critical)"
    if (
        baseline.get("swap_used_mb") is not None
        and snap["swap_used_mb"] - baseline["swap_used_mb"] >= SWAP_DRIFT_CEIL_MB
    ):
        drift = snap["swap_used_mb"] - baseline["swap_used_mb"]
        return False, f"swap drift {drift:.0f}MB > ceiling {SWAP_DRIFT_CEIL_MB}MB"
    if (
        snap["mps_alloc_mb"] is not None
        and baseline.get("mps_alloc_mb") is not None
        and snap["mps_alloc_mb"] - baseline["mps_alloc_mb"] > MPS_DRIFT_CEIL_MB
    ):
        drift = snap["mps_alloc_mb"] - baseline["mps_alloc_mb"]
        return False, f"MPS alloc drift {drift:.0f}MB > ceiling {MPS_DRIFT_CEIL_MB}MB"
    return True, "ok"


def preload_memory_gate() -> tuple[bool, str]:
    """Require PRELOAD_READINGS stable healthy readings before model load.
    Historical swap alone is NOT a rejection reason (see DESIGN_DECISION)."""
    for i in range(1, PRELOAD_READINGS + 1):
        if i > 1:
            time.sleep(PRELOAD_READING_GAP_SEC)
        snap = check_memory()
        log_memory(f"pre_gate_{i}of{PRELOAD_READINGS}", snap)
        if snap["free_pct"] is None or snap["swap_used_mb"] is None or snap["pressure_level"] is None:
            return False, f"reading {i}: telemetry unreadable -- fail-closed"
        if snap["free_pct"] < PRELOAD_FREE_PCT_MIN:
            return False, f"reading {i}: free_pct {snap['free_pct']}% < required {PRELOAD_FREE_PCT_MIN}%"
        if snap["pressure_level"] != 1:
            return False, f"reading {i}: memory pressure level {snap['pressure_level']} (not normal)"
    return True, "ok"


# --------------------------------------------------------------------------- #
# Data loading / matched-pair construction (pure Python + json; no ML libs)
# --------------------------------------------------------------------------- #


def load_week2_selected_items() -> list[dict]:
    if not WEEK2_SELECTED_ITEMS_JSONL.exists():
        print(
            f"NOT AVAILABLE FROM CURRENT FILES: missing {WEEK2_SELECTED_ITEMS_JSONL}. "
            f"Week 3 reuses Week 2's validated Age selection verbatim and will not "
            f"re-derive it.",
            file=sys.stderr,
        )
        sys.exit(1)
    rows = [json.loads(line) for line in WEEK2_SELECTED_ITEMS_JSONL.read_text().splitlines() if line]
    order_key: dict[str, int] = {}
    cond_order = {("ambig", "neg"): 0, ("ambig", "nonneg"): 1, ("disambig", "neg"): 2, ("disambig", "nonneg"): 3}
    for r in rows:
        if r["base_item_id"] not in order_key:
            order_key[r["base_item_id"]] = len(order_key)
    rows.sort(key=lambda r: (order_key[r["base_item_id"]], cond_order[(r["context_condition"], r["question_polarity"])]))
    return rows


def base_items_in_order(rows: list[dict]) -> list[str]:
    seen: list[str] = []
    for r in rows:
        if r["base_item_id"] not in seen:
            seen.append(r["base_item_id"])
    return seen


def build_prompt(fmt: str, row: dict) -> str:
    if fmt == "multiple_choice":
        return PROMPT_TEMPLATE_MC.format(
            context=row["context"], question=row["question"], ans0=row["ans0"], ans1=row["ans1"], ans2=row["ans2"]
        )
    if fmt == "open_ended":
        return PROMPT_TEMPLATE_OPEN.format(context=row["context"], question=row["question"])
    raise ValueError(f"unknown prompt_format {fmt!r}")


def pair_id_for(row: dict) -> str:
    return f"{row['base_item_id']}__{row['context_condition']}__{row['question_polarity']}"


def build_planned_prompts(rows_scope: list[dict]) -> list[dict]:
    """Expand each Week 2 matched BBQ row into its MC + open-ended pair.
    Does not touch torch/transformers/jlens; pure string formatting."""
    planned: list[dict] = []
    for row in rows_scope:
        pid = pair_id_for(row)
        for fmt in FORMATS:
            planned.append(
                {
                    "category": row["category"],
                    "base_item_id": row["base_item_id"],
                    "example_id": row["example_id"],
                    "pair_id": pid,
                    "context_condition": row["context_condition"],
                    "question_polarity": row["question_polarity"],
                    "prompt_format": fmt,
                    "prompt": build_prompt(fmt, row),
                    "question": row["question"],
                    "stereotype_group": row["role_stereotype_group_label"],
                    "counter_group": row["role_counter_group_label"],
                    "stereotype_target_text": row["role_stereotype_target_text"],
                    "counter_target_text": row["role_counter_target_text"],
                    "unknown_answer_key": row["role_unknown_key"],
                    "correct_answer_label": row["label"],
                    "stereotype_answer_key": row["role_stereotype_key"],
                    "counter_answer_key": row["role_counter_key"],
                }
            )
    return planned


def write_planned_prompts(planned: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PLANNED_PROMPTS.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for p in planned:
            f.write(json.dumps(p) + "\n")
    tmp.replace(PLANNED_PROMPTS)


# --------------------------------------------------------------------------- #
# Checkpointing (ported from Week 2; checkpoint name now also carries
# prompt_format so MC and open-ended results never collide)
# --------------------------------------------------------------------------- #


def checkpoint_path(planned_row: dict) -> Path:
    name = (
        f"{planned_row['base_item_id']}__{planned_row['context_condition']}__"
        f"{planned_row['question_polarity']}__{planned_row['prompt_format']}.csv"
    )
    return CKPT_DIR / name


def checkpoint_is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != CSV_FIELDS:
                return False
            return any(True for _ in reader)
    except Exception:
        return False


def write_checkpoint_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def mark_completed(planned_row: dict, ckpt: Path) -> None:
    completed = []
    if COMPLETED_JSON.exists():
        completed = json.loads(COMPLETED_JSON.read_text())
    key = (planned_row["base_item_id"], planned_row["context_condition"], planned_row["question_polarity"], planned_row["prompt_format"])
    entry = {
        "category": planned_row["category"],
        "base_item_id": planned_row["base_item_id"],
        "pair_id": planned_row["pair_id"],
        "context_condition": planned_row["context_condition"],
        "question_polarity": planned_row["question_polarity"],
        "prompt_format": planned_row["prompt_format"],
        "checkpoint_path": str(ckpt),
    }
    completed = [
        c
        for c in completed
        if (c["base_item_id"], c["context_condition"], c["question_polarity"], c["prompt_format"]) != key
    ]
    completed.append(entry)
    COMPLETED_JSON.write_text(json.dumps(completed, indent=2))


def is_already_done(planned_row: dict) -> bool:
    ckpt = checkpoint_path(planned_row)
    if not checkpoint_is_valid(ckpt):
        return False
    if not COMPLETED_JSON.exists():
        return False
    completed = json.loads(COMPLETED_JSON.read_text())
    key = (planned_row["base_item_id"], planned_row["context_condition"], planned_row["question_polarity"], planned_row["prompt_format"])
    return any(
        (c["base_item_id"], c["context_condition"], c["question_polarity"], c["prompt_format"]) == key
        for c in completed
    )


def combine_checkpoints() -> int:
    all_rows = []
    for ckpt in sorted(CKPT_DIR.glob("*.csv")):
        if ckpt.name.endswith(".tmp"):
            continue
        with open(ckpt, newline="") as f:
            reader = csv.DictReader(f)
            all_rows.extend(reader)
    if not all_rows:
        return 0
    with open(RAW_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    return len(all_rows)


def write_recovery_state(extra: dict) -> None:
    state = {
        "branch": "melody-week2",  # actual current git branch at authoring time (VERIFIED via git status)
        "category": "Age",
        "model": MODEL_NAME,
        "lens_repo": LENS_REPO,
        "lens_revision": LENS_REVISION,
        "lens_file": LENS_FILE,
        "runner_path": str(Path(__file__).resolve()),
        "checkpoint_dir": str(CKPT_DIR),
        "manifest_path": str(COMPLETED_JSON),
        "raw_csv_path": str(RAW_CSV),
        "memory_log_path": str(MEMORY_LOG),
        "notebook_path": str(WEEK3_DIR / "submissions" / "MelodyResearch_week_3.ipynb"),
        "rank_gap_definition": "rank_gap = counter_rank - stereotype_rank (positive = stereotype ranks stronger) -- inherited verbatim from Week 2",
        "top_k": TOP_K,
        "prompt_formats": FORMATS,
        "timestamp": time.time(),
    }
    state.update(extra)
    RECOVERY_STATE.write_text(json.dumps(state, indent=2))


# --------------------------------------------------------------------------- #
# Model / lens setup (imports ML stack -- only ever called after BOTH
# authorization flags are verified AND pre-load safety gates pass)
# --------------------------------------------------------------------------- #


def load_model_and_lens():
    import torch
    import transformers

    import jlens

    jlens.configure_logging()

    print(f"Loading model {MODEL_NAME} on MPS ...", file=sys.stderr)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to("mps")
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf_model, tokenizer)
    print(model, file=sys.stderr)

    print("Loading released pre-fitted lens ...", file=sys.stderr)
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION)
    print(lens, file=sys.stderr)

    assert lens.d_model == model.d_model, (
        f"lens/model d_model mismatch: lens.d_model={lens.d_model} vs model.d_model={model.d_model}"
    )
    assert MODEL_NAME == "Qwen/Qwen3.5-0.8B", "hf_model_name must be Qwen/Qwen3.5-0.8B"
    max_layer = max(lens.source_layers)
    assert max_layer < model.n_layers, (
        f"lens fitted layers (max={max_layer}) incompatible with model.n_layers={model.n_layers}"
    )
    print(
        f"VERIFIED: model={MODEL_NAME} n_layers={model.n_layers} d_model={model.d_model}; "
        f"lens d_model={lens.d_model} n_prompts={lens.n_prompts} "
        f"source_layers=[{lens.source_layers[0]}..{lens.source_layers[-1]}] "
        f"({len(lens.source_layers)} layers) file={LENS_FILE} revision={LENS_REVISION}",
        file=sys.stderr,
    )
    return model, tokenizer, lens


# --------------------------------------------------------------------------- #
# Prompt processing (ported from Week 2's process_row; source position is
# re-verified per prompt rather than assumed, since the open-ended prompt is
# a structurally new format never empirically checked in Week 2)
# --------------------------------------------------------------------------- #


def constituent_ranks(tokenizer, target_text: str, logits_1d, mask) -> tuple[list[str], list[int]]:
    from jlens import vis
    import torch

    ids = tokenizer(" " + target_text, add_special_tokens=False).input_ids
    ids = [i for i in ids if bool(mask[i])]
    if not ids:
        return [], []
    ids_t = torch.tensor(ids, dtype=torch.long)
    ranks = vis._ranks_of(logits_1d.unsqueeze(0), ids_t.unsqueeze(0))[0]
    toks = [tokenizer.decode([i], clean_up_tokenization_spaces=False) for i in ids]
    return toks, [int(r) for r in ranks.tolist()]


def process_planned_row(model, tokenizer, lens, planned_row: dict) -> list[dict]:
    import torch
    from jlens import vis

    prompt = planned_row["prompt"]
    input_ids = model.encode(prompt)
    seq_len = int(input_ids.shape[1])
    source_pos = seq_len - 1
    source_token = tokenizer.decode([int(input_ids[0, source_pos])], clean_up_tokenization_spaces=False)
    if source_token.strip() != ":":
        # DESIGN_DECISION (Week 3): unlike Week 2, we assert rather than assume
        # the "Answer:" convention holds for the new open_ended format. If it
        # does not, we do not silently redefine analysis_position -- we flag it.
        print(
            f"WARNING: analysis_position source_token={source_token!r} is not ':' "
            f"for prompt_format={planned_row['prompt_format']!r} pair_id={planned_row['pair_id']!r}. "
            f"UNVERIFIED convention for this row -- recorded as-is, not corrected.",
            file=sys.stderr,
        )

    lens_logits, model_logits, _ = lens.apply(model, prompt, layers=lens.source_layers, positions=[source_pos])

    final_layer = model.n_layers - 1
    vocab_size = model_logits.shape[-1]
    mask = vis._meaningful_token_mask(tokenizer, vocab_size, model_logits.device)

    rows: list[dict] = []

    def row_for(layer_label, readout: str, logits_2d) -> None:
        logits_1d = logits_2d[0]
        masked = logits_1d.masked_fill(~mask, float("-inf"))
        top_idx = masked.topk(TOP_K).indices
        top_ranks = vis._ranks_of(logits_1d.unsqueeze(0), top_idx.unsqueeze(0))[0]
        top_tokens = [tokenizer.decode([int(t)], clean_up_tokenization_spaces=False) for t in top_idx]

        s_toks, s_ranks = constituent_ranks(tokenizer, planned_row["stereotype_target_text"], logits_1d, mask)
        c_toks, c_ranks = constituent_ranks(tokenizer, planned_row["counter_target_text"], logits_1d, mask)
        stereo_rank = min(s_ranks) if s_ranks else None
        counter_rank = min(c_ranks) if c_ranks else None
        rank_gap = (counter_rank - stereo_rank) if (stereo_rank is not None and counter_rank is not None) else None

        rows.append(
            {
                "category": planned_row["category"],
                "base_item_id": planned_row["base_item_id"],
                "example_id": planned_row["example_id"],
                "pair_id": planned_row["pair_id"],
                "context_condition": planned_row["context_condition"],
                "question_polarity": planned_row["question_polarity"],
                "prompt_format": planned_row["prompt_format"],
                "prompt": prompt,
                "question": planned_row["question"],
                "layer": layer_label,
                "readout": readout,
                "seq_len": seq_len,
                "analysis_position": source_pos,
                "source_token": source_token,
                "top20_tokens": json.dumps(top_tokens),
                "top20_ranks": json.dumps([int(r) for r in top_ranks.tolist()]),
                "stereotype_group": planned_row["stereotype_group"],
                "counter_group": planned_row["counter_group"],
                "stereotype_target_text": planned_row["stereotype_target_text"],
                "counter_target_text": planned_row["counter_target_text"],
                "stereotype_constituent_tokens": json.dumps(s_toks),
                "stereotype_constituent_ranks": json.dumps(s_ranks),
                "counter_constituent_tokens": json.dumps(c_toks),
                "counter_constituent_ranks": json.dumps(c_ranks),
                "stereotype_rank": stereo_rank,
                "counter_rank": counter_rank,
                "rank_gap": rank_gap,
                "unknown_answer_key": planned_row["unknown_answer_key"],
                "correct_answer_label": planned_row["correct_answer_label"],
                "stereotype_answer_key": planned_row["stereotype_answer_key"],
                "counter_answer_key": planned_row["counter_answer_key"],
            }
        )

    for layer in lens.source_layers:
        row_for(layer, "jacobian_lens", lens_logits[layer])
    row_for(final_layer, "model_output", model_logits)

    del lens_logits, model_logits, mask, input_ids
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepare-only", action="store_true", help="Default behavior: build/validate planned prompt pairs, no model load. (default even if omitted)")
    parser.add_argument("--sanity-only", action="store_true", help="If inference is authorized, restrict to exactly ONE matched base item (8 prompts).")
    parser.add_argument("--max-base-items", type=int, default=HARD_CAP_BASE_ITEMS, help="Hard cap on base items this run (max 10).")
    parser.add_argument("--run-inference", action="store_true", help="Required (together with --i-confirm-no-heavy-experiment) to allow model loading.")
    parser.add_argument("--i-confirm-no-heavy-experiment", action="store_true", help="Required (together with --run-inference) to allow model loading.")
    args = parser.parse_args()
    max_base_items = min(args.max_base_items, HARD_CAP_BASE_ITEMS)

    inference_authorized = args.run_inference and args.i_confirm_no_heavy_experiment
    if args.run_inference != args.i_confirm_no_heavy_experiment:
        print(
            "NOTE: only one of --run-inference / --i-confirm-no-heavy-experiment was given. "
            "BOTH are required to authorize model loading; falling back to prepare-only.",
            file=sys.stderr,
        )

    # ---- Always-safe part: build the matched-pair plan (no ML imports) ----
    all_rows = load_week2_selected_items()
    base_order = base_items_in_order(all_rows)
    if args.sanity_only:
        scope_base_items = base_order[:1]
    else:
        scope_base_items = base_order[:max_base_items]
    rows_scope = [r for r in all_rows if r["base_item_id"] in scope_base_items]
    planned = build_planned_prompts(rows_scope)
    write_planned_prompts(planned)

    n_base = len(scope_base_items)
    n_evals = len(planned)  # = n_base * CONDITIONS_PER_BASE_ITEM * len(FORMATS)
    print(
        f"PLANNED: {n_base} base item(s) x {CONDITIONS_PER_BASE_ITEM} condition(s) x "
        f"{len(FORMATS)} prompt format(s) = {n_evals} model evaluation(s) "
        f"(1 forward pass each; layers come from a single lens.apply() call). "
        f"Written to {PLANNED_PROMPTS}."
    )

    if not inference_authorized:
        print(
            "PREPARE-ONLY MODE: exiting before any model/runtime import or load. "
            "Re-run with --run-inference --i-confirm-no-heavy-experiment to authorize inference "
            "(and add --sanity-only to additionally cap the run to one base item)."
        )
        sys.exit(0)

    # ---- From here on, inference is authorized: run pre-load safety gates ----
    check_no_other_week3_process()
    acquire_lock()
    try:
        run_inference(planned, n_base, n_evals)
    finally:
        release_lock()


def run_inference(planned: list[dict], n_base: int, n_evals: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    ok, reason = preload_memory_gate()
    if not ok:
        print(f"MEMORY SAFETY STOP: unsafe BEFORE model load ({reason}); aborting without loading model.")
        write_recovery_state(
            {"needs_finalization": True, "stopped_at": "pre_model_load", "stopped_reason": reason, "completed_prompts": 0}
        )
        sys.exit(2)

    pending = [r for r in planned if not is_already_done(r)]
    print(f"Planned model-evaluation count for this authorized run: {n_evals} across {n_base} base item(s). {len(pending)} still pending.")
    if not pending:
        print("All requested prompts already have valid checkpoints. Nothing to do.")
        n_rows = combine_checkpoints()
        print(f"{n_rows} total rows in {RAW_CSV}.")
        return

    pre_load = check_memory()
    log_memory("pre_model_load", pre_load)
    ok, reason = memory_is_safe(pre_load, pre_load)
    if not ok:
        print(f"MEMORY SAFETY STOP: unsafe immediately BEFORE model load ({reason}); aborting without loading model.")
        write_recovery_state(
            {"needs_finalization": True, "stopped_at": "pre_model_load", "stopped_reason": reason, "completed_prompts": 0}
        )
        sys.exit(2)
    BASELINE_SWAP_FILE.write_text(f"{pre_load['swap_used_mb']}\n")
    print(f"BASELINE_SWAP_MB={pre_load['swap_used_mb']}", file=sys.stderr)

    model, tokenizer, lens = load_model_and_lens()
    import torch

    post_load = check_memory(torch)
    log_memory("post_model_load", post_load)
    ok, reason = memory_is_safe(post_load, pre_load)
    if not ok:
        print(f"MEMORY SAFETY STOP after model load: {reason}")
        write_recovery_state(
            {"needs_finalization": True, "stopped_at": "post_model_load", "stopped_reason": reason, "completed_prompts": 0}
        )
        sys.exit(2)

    # MPS baseline = post-load; swap baseline stays BASELINE_SWAP_MB (pre-load) so
    # swap growth caused by the model load itself is not forgotten.
    baseline = {**post_load, "swap_used_mb": pre_load["swap_used_mb"]}
    done_base_items: set[str] = set()
    n_completed_this_run = 0
    stopped_reason = None

    for planned_row in pending:
        ckpt = checkpoint_path(planned_row)
        tag = f"{planned_row['base_item_id']}/{planned_row['context_condition']}/{planned_row['question_polarity']}/{planned_row['prompt_format']}"
        print(f"--- {tag} ---", file=sys.stderr)
        t0 = time.time()
        try:
            out_rows = process_planned_row(model, tokenizer, lens, planned_row)
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR processing {tag}: {exc!r}; stopping (not marking complete).", file=sys.stderr)
            stopped_reason = f"exception on {tag}: {exc!r}"
            break
        write_checkpoint_atomic(ckpt, out_rows)
        assert checkpoint_is_valid(ckpt), f"checkpoint write verification failed: {ckpt}"
        mark_completed(planned_row, ckpt)
        n_completed_this_run += 1
        dt = time.time() - t0
        print(f"    {len(out_rows)} rows -> {ckpt.name} in {dt:.1f}s", file=sys.stderr)

        del out_rows
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        snap = check_memory(torch)
        log_memory(f"after_{ckpt.stem}", snap)
        ok, reason = memory_is_safe(snap, baseline)
        if not ok:
            print(f"MEMORY SAFETY STOP: {reason}")
            stopped_reason = reason
            break

        done_base_items.add(planned_row["base_item_id"])
        n_done_base = len(done_base_items)
        rows_for_base = [r for r in planned if r["base_item_id"] == planned_row["base_item_id"]]
        base_fully_done = all(is_already_done(r) for r in rows_for_base)
        if base_fully_done and n_done_base in STAGE_BOUNDARIES:
            print(f"Stage boundary reached: {n_done_base} base item(s) complete. Re-checking memory ...")
            ok, reason = memory_is_safe(snap, pre_load)
            if not ok:
                print(f"MEMORY SAFETY STOP at stage boundary {n_done_base}: {reason}")
                stopped_reason = reason
                break

    n_rows = combine_checkpoints()
    n_manifest = len(json.loads(COMPLETED_JSON.read_text())) if COMPLETED_JSON.exists() else 0
    print(f"Done. {n_manifest} prompt(s) total completed; {n_rows} total rows in {RAW_CSV}.")

    remaining = [r for r in planned if not is_already_done(r)]
    write_recovery_state(
        {
            "needs_finalization": stopped_reason is not None,
            "stopped_reason": stopped_reason,
            "completed_prompts": n_manifest,
            "remaining_prompts": len(remaining),
        }
    )
    if stopped_reason:
        print(f"Stopped early: {stopped_reason}. State saved to {RECOVERY_STATE}.")


if __name__ == "__main__":
    main()
