# Week 3 Provenance

## Final run update (2026-09-18) -- supersedes the pre-run notes below

One full run was executed:
`run_week3_experiment.py --max-base-items 10 --run-inference --i-confirm-no-heavy-experiment`
(exit 0): 10 base items x 4 conditions x 2 formats = 80 prompt evaluations, 1,920 rows
(24 per evaluation: 23 J-Lens layers + `model_output`), on `Qwen/Qwen3.5-0.8B` with the
released `qwen-n1000` lens, Apple MPS. `finalize_week3.py` then built summaries/figures and
executed the notebook (no model inference).

- **Memory-gate change (safety logic only, no scientific logic):** the absolute pre-load swap
  ceiling (`PRELOAD_SWAP_ABS_CEIL_MB = 4096`, now kept only as `LEGACY_PRELOAD_SWAP_ABS_CEIL_MB`)
  refused to load because ~5030 MB of historical swap remained after a prior experiment exited.
  It was replaced by: 3 pre-load readings ~25 s apart with >=60% free and normal pressure level;
  `BASELINE_SWAP_MB` recorded immediately before load; stop at free <30%, swap growth >=+1500 MB over
  baseline (runner + `week3_watchdog.sh`), swap +750 MB within ~60 s, critical pressure level, MPS
  drift >2048 MB, unreadable telemetry. Observed: pre-load free 71/71/72%, baseline swap 5030.06 MB,
  swap delta 0 MB, min free 58%, pressure level 1, MPS 1507.9 MB constant, no stop triggered.
- **Resolved:** `source_pos` lands on `':'` for all 1,920 rows including all open-ended rows (0 warnings).
- **Audit fix:** `audit_results.py` keyed duplicates on evaluation only (each evaluation has 24 rows);
  now keyed on (evaluation, layer, readout) and also checks 80 distinct evaluations.
- Week 3's MC arm reproduces Week 2 exactly (920 rows, max |rank_gap diff| = 0), so it is a consistency
  check rather than an independent replication.

The sections below are the original pre-run (static development) record and are kept for provenance;
statements marked `NOT YET RUN` / "not executed" / "do not yet exist" are superseded by this update.

## Verified facts (Week 2 recovered configuration)

| Fact | Status | Source file | Key | Note |
|---|---|---|---|---|
| Model | VERIFIED_FROM_LOCAL_SOURCE | `Week 2/scripts/run_week2_experiment.py:70` | `MODEL_NAME` | `Qwen/Qwen3.5-0.8B` |
| Lens repo | VERIFIED_FROM_LOCAL_SOURCE | same file:71 | `LENS_REPO` | `neuronpedia/jacobian-lens` |
| Lens revision | VERIFIED_FROM_LOCAL_SOURCE | same file:72 | `LENS_REVISION` | `qwen-n1000` |
| Lens file | VERIFIED_FROM_LOCAL_SOURCE | same file:73 | `LENS_FILE` | `qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt` |
| Device / dtype | VERIFIED_FROM_LOCAL_SOURCE | same file:406 | `.to("mps")`, `dtype=torch.bfloat16` | Apple MPS, bf16 |
| BBQ category | VERIFIED_FROM_LOCAL_SOURCE | `week2_recovery_state.json` | `"category": "Age"` | |
| Base items | VERIFIED_FROM_LOCAL_SOURCE | `Week 2/data/selection_manifest.json` | `n_base_items` | 10 items, 40 rows (4 conditions each) |
| top_k | VERIFIED_FROM_LOCAL_SOURCE | `run_week2_experiment.py:75` | `TOP_K = 20` | |
| Prompt template (MC) | VERIFIED_FROM_LOCAL_SOURCE | `run_week2_experiment.py:115` | `PROMPT_TEMPLATE` | `{context}\n{question}\n(a)...(c)...\nAnswer:` |
| Analysis position | VERIFIED_FROM_LOCAL_SOURCE | `run_week2_experiment.py:471` | `source_pos = seq_len - 1` | last token, comment-verified `':'` of `Answer:` |
| Multi-token target rule | VERIFIED_FROM_LOCAL_SOURCE | `run_week2_experiment.py:445-459` | `constituent_ranks()` | `" " + target_text`, meaningful-token filter, `min(rank)` |
| rank_gap definition | VERIFIED_FROM_LOCAL_SOURCE | `run_week2_experiment.py:498` + `week2_recovery_state.json` | `rank_gap` | `counter_rank - stereotype_rank`; positive = stereotype ranks stronger |
| Sequential, 1 model instance | VERIFIED_FROM_LOCAL_SOURCE | `run_week2_experiment.py` (whole `run()` loop) | -- | one `load_model_and_lens()` call, prompts processed in a `for` loop |
| Checkpoint/resume | VERIFIED_FROM_LOCAL_SOURCE | `run_week2_experiment.py:288-367` | `checkpoint_path`, `is_already_done`, `combine_checkpoints` | atomic `.tmp`→rename writes |
| PID/run lock | VERIFIED_FROM_LOCAL_SOURCE | `run_week2_experiment.py:131-185` | `acquire_lock`, `check_no_other_qwen_process` | |
| Memory monitoring | VERIFIED_FROM_LOCAL_SOURCE | `run_week2_experiment.py:193-247` | `check_memory`, `memory_is_safe` | `memory_pressure`, `sysctl vm.swapusage`, MPS alloc |
| No locally fitted lens | VERIFIED_FROM_LOCAL_SOURCE | `run_week2_experiment.py:412` | `JacobianLens.from_pretrained` | never `.fit()`/`.train()`/`.refit()` (grepped, absent) |
| jlens API surface used | VERIFIED_FROM_LOCAL_SOURCE | `~/Desktop/JLens-project/jacobian-lens/jlens/{lens,vis,hf}.py` | `from_hf`, `JacobianLens.from_pretrained`, `.apply`, `_meaningful_token_mask`, `_ranks_of` | confirmed present in local `jlens` source, not invented |
| Stereotype/counter role derivation | VERIFIED_FROM_LOCAL_SOURCE | `Week 2/scripts/select_items.py:76-101` | `derive_roles()` | from BBQ `answer_info` + `additional_metadata.stereotyped_groups`; items skipped if role not unique |

## Week 3 README requirements used

| Requirement | Status | Source |
|---|---|---|
| Assignment text | VERIFIED_FROM_LOCAL_SOURCE | `upstream/master:Week 3/README.md` (fetched via `git fetch upstream`; not yet present on any local branch) |
| Option chosen: drop multiple-choice | VERIFIED_FROM_LOCAL_SOURCE | Week 3 README, "Option 1 → Suggested approaches → 1. Drop the multiple-choice" |
| Option 2 (steering) explicitly NOT used | DESIGN_DECISION | Per user's explicit task instruction, not the README (README allows either option) |

## New Week 3 design decisions

| Decision | Status | Rationale |
|---|---|---|
| Open-ended template = `{context}\n{question}\nAnswer:` | DESIGN_DECISION | Minimal transformation of the Week 2 MC template: only the `(a)/(b)/(c)` block is removed; context/question text untouched |
| `--sanity-only` = 1 base item × 4 conditions × 2 formats = 8 evals | DESIGN_DECISION | Task instructions explicitly define Week 3 sanity/Stage 0 as "one matched base item," which differs from Week 2's stricter 1-row sanity; documented as an intentional deviation |
| Default mode with no flags = prepare-only, no ML import | DESIGN_DECISION | Explicit Week 3 safety requirement, stricter than Week 2 (which had no separate prepare-only mode) |
| Dual-flag inference authorization (`--run-inference` AND `--i-confirm-no-heavy-experiment`) | DESIGN_DECISION | Explicit Week 3 safety requirement |
| Fail-closed on unreadable `memory_pressure`/`sysctl` output | DESIGN_DECISION | Week 2's `memory_is_safe` silently skips a check when a reading is `None` (fails open); Week 3 treats an unreadable reading as unsafe |
| `PRELOAD_SWAP_ABS_CEIL_MB = 4096`, `PRELOAD_FREE_PCT_FLOOR = 25` | DESIGN_DECISION | Absolute pre-load gates (not just drift-from-baseline) to account for a different experiment already consuming memory before Week 3 starts |
| Per-prompt `source_token == ':'` assertion → warning, not silent correction | DESIGN_DECISION | The `Answer:`-ending convention was only empirically verified for Week 2's MC template; the new open-ended template's tokenization is `UNVERIFIED` until an actual run confirms it |

## UNVERIFIED / NOT AVAILABLE FROM CURRENT FILES

- Whether `source_pos = seq_len - 1` is truly the last token before generation for the **open-ended** prompt format specifically (`{context}\n{question}\nAnswer:`) — only verified for the MC format in Week 2. The Week 3 runner asserts and logs a warning if this doesn't hold per-row; it is not assumed.
- Exact tokenizer behavior (BPE splits) for the open-ended prompt's final tokens — `NOT AVAILABLE FROM CURRENT FILES` without running the tokenizer, which this session does not do.
- Whether all 10 Week 2 Age base items remain scientifically valid matched pairs once choices are dropped (e.g., a question that only makes sense with an answer list present) — not reviewed item-by-item in this session; flagged for manual review before an authorized run.
- Any actual Week 3 results (all result files listed below do not yet exist).

## Runtime checks still required (before any authorized inference)

1. `memory_pressure` free% and `sysctl vm.swapusage` at time of run (script checks automatically, fail-closed).
2. Confirm no other Week 3 runner PID lock is held.
3. Confirm the other, currently-running experiment on this Mac has finished or is known not to compete for MPS/unified memory — this script does **not** attempt to detect or judge that experiment; the user must decide.
4. Actually run `--sanity-only` (1 base item, 8 evaluations) and inspect output before any larger run.

## Files created this session

- `Week 3/scripts/run_week3_experiment.py` (prepare-only by default; dual-flag-gated inference)
- `Week 3/scripts/finalize_week3.py` (pandas/matplotlib only, no ML imports; not executed this session)
- `Week 3/submissions/MelodyResearch_week_3.ipynb` (no executed cells; result-loading cells check for file existence and state "Results will be populated after the experiment is executed." when absent)
- `Week 3/results/` and `Week 3/results/checkpoints/` (empty directories, created for future runs)
- `Week 3/PROVENANCE.md` (this file)

`Week 2/` was not modified. No inference, model load, MPS work, or notebook execution occurred in this session.
