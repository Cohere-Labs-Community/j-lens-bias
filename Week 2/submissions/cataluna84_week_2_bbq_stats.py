#!/usr/bin/env python3
"""Week 2: Jacobian-lens bias readout on the BBQ dataset.

Research question: does the lens readout surface stereotype-congruent tokens
more strongly than counter-stereotype tokens, and does the signal persist in
ambiguous contexts where the correct answer is "unknown"?

Design:
- Stratified sample: BBQ items come in matched blocks of 4 (neg-ambig,
  neg-disambig, nonneg-ambig, nonneg-disambig) sharing one option ordering.
  Sampling blocks keeps the 2x2 split balanced by construction.
- ``target_loc`` from ``supplemental/additional_metadata.csv`` is already
  polarity-corrected (verified structurally: within a block it is never the
  unknown option, is constant within polarity, and flips between neg and
  nonneg). ``stereo_loc = target_loc``; the counter option is the remaining
  non-unknown slot.
- Two prompt formats per item:
    arc:  "{context} {question}\\n(a) .. (b) .. (c) ..\\nAnswer: ("  -> track
          the letter tokens a/b/c at the answer slot.
    free: "{context} {question}\\nAnswer:" -> track the first token at which
          the stereo and counter option tokenizations differ (e.g.
          " grandfather" vs " grandson"; " woman" vs " man" for
          "The Black woman" vs "The Black man"), plus the unknown option's
          first content token.
- One ``lens.apply(positions=None)`` per item x format yields the full
  layer x position grid; per layer we record final-position ranks of the
  tracked tokens, their best rank over positions, ranks at the positions
  where the group words are mentioned in the context, and the word-like
  top-15 at the answer slot (qualitative).

Modes: ``collect`` (resumable record writer), ``fidelity`` (int8-GPU vs
CPU-bf16 rank agreement on a few items), ``summarize`` (tables + plots).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import transformers

import jlens
from jlens.vis import _meaningful_token_mask, _ranks_of

CATEGORIES = [
    "Age", "Disability_status", "Gender_identity", "Nationality",
    "Physical_appearance", "Race_ethnicity", "Race_x_SES", "Race_x_gender",
    "Religion", "SES", "Sexual_orientation",
]

TOP_K = 15
# Positions 0-15 are outside the lens fitting distribution
# (SKIP_FIRST_N_POSITIONS); best-rank metrics are reported both ways.
FIT_SKIP = 16


# --------------------------------------------------------------------------
# Dataset loading and labeling
# --------------------------------------------------------------------------

def load_metadata(bbq_dir: Path) -> dict:
    import csv

    md = {}
    with (bbq_dir / "supplemental" / "additional_metadata.csv").open() as fh:
        for r in csv.DictReader(fh):
            md[(r["category"], int(r["example_id"]))] = r
    return md


def load_blocks(cat: str, md: dict, data_dir: Path) -> list[list[dict]]:
    """Return matched blocks of 4 items with stereo/counter/unknown labels.

    Block layout (verified for all 11 categories): neg-ambig, neg-disambig,
    nonneg-ambig, nonneg-disambig, sharing one option ordering.
    """
    rows = [json.loads(line) for line in (data_dir / f"{cat}.jsonl").open()]
    rows.sort(key=lambda r: r["example_id"])
    blocks, i = [], 0
    while i + 4 <= len(rows):
        blk = rows[i:i + 4]
        ok = (
            [b["question_polarity"] for b in blk] == ["neg", "neg", "nonneg", "nonneg"]
            and [b["context_condition"] for b in blk]
            == ["ambig", "disambig", "ambig", "disambig"]
            and len({(b["ans0"], b["ans1"], b["ans2"]) for b in blk}) == 1
        )
        if not ok:
            i += 1
            continue
        recs = []
        for b in blk:
            m = md.get((cat, b["example_id"]))
            if not m or m["target_loc"] in ("NA", ""):
                ok = False
                break
            tl = int(m["target_loc"])
            unk = next(
                j for j in range(3) if b["answer_info"][f"ans{j}"][1] == "unknown"
            )
            if tl == unk:
                ok = False
                break
            counter = next(j for j in range(3) if j not in (tl, unk))
            recs.append(
                {
                    "example_id": b["example_id"],
                    "polarity": b["question_polarity"],
                    "condition": b["context_condition"],
                    "label": b["label"],
                    "context": b["context"],
                    "question": b["question"],
                    "options": [b["ans0"], b["ans1"], b["ans2"]],
                    "stereo_loc": tl,
                    "counter_loc": counter,
                    "unknown_loc": unk,
                    "stereotyped_groups": b["additional_metadata"].get(
                        "stereotyped_groups", []
                    ),
                }
            )
        tls = [r["stereo_loc"] for r in recs]
        if ok and not (tls[0] == tls[1] and tls[2] == tls[3] and tls[0] != tls[2]):
            ok = False
        if ok:
            blocks.append(recs)
        i += 4
    return blocks


def sample_items(
    bbq_dir: Path, categories: list[str], blocks_per_cat: int, seed: int
) -> list[dict]:
    """Flatten sampled blocks into item dicts with a stable item key."""
    md = load_metadata(bbq_dir)
    rng = random.Random(seed)
    items = []
    for cat in categories:
        blocks = load_blocks(cat, md, bbq_dir / "data")
        rng.shuffle(blocks)
        for blk in blocks[:blocks_per_cat]:
            for rec in blk:
                rec["category"] = cat
                rec["item_key"] = f"{cat}:{rec['example_id']}"
                items.append(rec)
    return items


# --------------------------------------------------------------------------
# Prompt construction and tracked tokens
# --------------------------------------------------------------------------

def content_ids(tokenizer, text: str) -> list[int]:
    """Leading-space tokenization of an answer string, minus a leading 'The'."""
    ids = tokenizer(" " + text.strip(), add_special_tokens=False).input_ids
    if ids and tokenizer.decode([ids[0]]).strip().lower() == "the":
        ids = ids[1:]
    return ids


def distinguishing_ids(s_ids: list[int], c_ids: list[int]) -> tuple[int, int, int]:
    """First index where the stereo/counter option tokenizations differ."""
    k = 0
    while k < min(len(s_ids), len(c_ids)) and s_ids[k] == c_ids[k]:
        k += 1
    if k == min(len(s_ids), len(c_ids)):  # one is a prefix of the other
        k = max(min(len(s_ids), len(c_ids)) - 1, 0)
    return s_ids[k], c_ids[k], k


def build_prompts(item: dict) -> dict:
    o = item["options"]
    return {
        "arc": (
            f"{item['context']} {item['question']}\n"
            f"(a) {o[0]} (b) {o[1]} (c) {o[2]}\nAnswer: ("
        ),
        "free": f"{item['context']} {item['question']}\nAnswer:",
    }


def tracked_tokens(tokenizer, item: dict) -> dict:
    """Per-format token ids whose ranks stand in for the three options.

    Returns ``{fmt: {"stereo": id, "counter": id, "unknown": id}}`` plus
    auxiliary info for the collision audit and entity-position lookup.
    """
    s_txt = item["options"][item["stereo_loc"]]
    c_txt = item["options"][item["counter_loc"]]
    u_txt = item["options"][item["unknown_loc"]]

    # ARC: after "Answer: (" the next token should be the bare letter.
    letter_ids = {
        ch: tokenizer(ch, add_special_tokens=False).input_ids[0]
        for ch in "abc"
    }
    arc = {
        "stereo": letter_ids["abc"[item["stereo_loc"]]],
        "counter": letter_ids["abc"[item["counter_loc"]]],
        "unknown": letter_ids["abc"[item["unknown_loc"]]],
    }

    s_ids, c_ids, u_ids = (
        content_ids(tokenizer, s_txt),
        content_ids(tokenizer, c_txt),
        content_ids(tokenizer, u_txt),
    )
    s_tok, c_tok, dist_idx = distinguishing_ids(s_ids, c_ids)
    free = {"stereo": s_tok, "counter": c_tok, "unknown": u_ids[0] if u_ids else -1}

    aux = {
        "s_content": s_ids,
        "c_content": c_ids,
        "dist_idx": dist_idx,
        "s_first": s_ids[0] if s_ids else -1,
        "c_first": c_ids[0] if c_ids else -1,
        "s_word": tokenizer.decode([s_ids[0]]).strip() if s_ids else "",
        "c_word": tokenizer.decode([c_ids[0]]).strip() if c_ids else "",
        "u_word": tokenizer.decode([u_ids[0]]).strip() if u_ids else "",
    }
    return {"arc": arc, "free": free, "aux": aux}


def entity_positions(
    tokenizer, prompt: str, context: str, input_ids: list[int], words: list[str]
) -> list[int]:
    """Token positions where any of ``words`` is mentioned in the context part.

    Uses offset mapping against the same tokenization the model saw; falls
    back to case-insensitive matching when the exact casing is absent.
    """
    enc = tokenizer(prompt, return_offsets_mapping=True)
    if enc.input_ids != input_ids:
        return []
    offs = enc.offset_mapping
    positions = set()
    for w in words:
        if not w or not re.search(r"\w", w):
            continue
        for flags in (0, re.IGNORECASE):
            hits = list(re.finditer(re.escape(w), context, flags))
            if hits:
                for h in hits:
                    for pos, (s, e) in enumerate(offs):
                        if s <= h.start() < e or (s >= h.start() and s < h.end()):
                            positions.add(pos)
                break
    return sorted(positions)


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------

def load_model_and_lens(args):
    print(f"loading {args.model} ...", flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    # RTX 2070 (WSL2 GPU 0) is 8 GB Turing. int8 fits; fp16 is the dtype the
    # card has tensor cores for. bf16 weights are ~9 GB and do not.
    if args.device == "cuda" and args.int8:
        torch.cuda.set_device(0)
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["lm_head", "embed_tokens"],
        )
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            dtype=torch.float16,
            device_map={"": 0},
        )
    else:
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16
        ).to(args.device)
    model = jlens.from_hf(hf_model, tokenizer)
    print(model, flush=True)

    lens = jlens.JacobianLens.from_pretrained(
        args.lens_repo, filename=args.lens_file, revision=args.lens_revision
    )
    print(lens, flush=True)
    return model, tokenizer, lens


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------

def collect(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"

    items = sample_items(
        Path(args.bbq_dir), args.categories, args.blocks_per_cat, args.seed
    )
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} items "
          f"({len(items) // 4} blocks) from {len(args.categories)} categories",
          flush=True)

    done = set()
    if records_path.exists() and args.resume:
        with records_path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                done.add((rec["item_key"], rec["format"]))
        print(f"resuming: {len(done)} item-formats already recorded", flush=True)

    model, tokenizer, lens = load_model_and_lens(args)
    final_layer = model.n_layers - 1
    mask = None
    t0 = time.time()
    n_run = 0

    fh = records_path.open("a", encoding="utf-8")
    for idx, item in enumerate(items):
        prompts = build_prompts(item)
        tracked = tracked_tokens(tokenizer, item)
        for fmt in args.formats:
            if (item["item_key"], fmt) in done:
                continue
            prompt = prompts[fmt]
            tk = tracked[fmt]
            lens_logits, model_logits, input_ids = lens.apply(
                model, prompt, positions=None, max_seq_len=args.max_seq_len
            )
            ids_list = input_ids[0].tolist()
            seq_len = len(ids_list)
            final = seq_len - 1

            if fmt == "arc":
                words = [tracked["aux"]["s_word"], tracked["aux"]["c_word"]]
            else:
                words = [
                    tracked["aux"]["s_word"],
                    tracked["aux"]["c_word"],
                    tokenizer.decode([tk["stereo"]]).strip(),
                    tokenizer.decode([tk["counter"]]).strip(),
                ]
            ent_pos = entity_positions(
                tokenizer, prompt, item["context"], ids_list, words
            )[:8]

            tracked_vec = torch.tensor(
                [tk["stereo"], tk["counter"], tk["unknown"]], dtype=torch.long
            )
            layer_logits = dict(lens_logits)
            layer_logits[final_layer] = model_logits

            layer_recs = []
            for layer, logits_cpu in sorted(layer_logits.items()):
                logits = logits_cpu.to(args.device)
                if mask is None:
                    mask = _meaningful_token_mask(
                        tokenizer, int(logits.shape[-1]), logits.device
                    )
                    print(
                        f"word-like vocab entries: {int(mask.sum())}", flush=True
                    )
                ranks_all = _ranks_of(logits, tracked_vec.to(logits.device))
                fr = ranks_all[final].tolist()

                best, best16 = [], []
                for col in range(3):
                    r = ranks_all[:, col]
                    b = int(r.min())
                    best.append([b, int(r.argmin())])
                    late = r[FIT_SKIP:] if seq_len > FIT_SKIP else r
                    best16.append([int(late.min()), int(late.argmin()) + (FIT_SKIP if seq_len > FIT_SKIP else 0)])

                ent = [
                    [p] + ranks_all[p].tolist() for p in ent_pos if p < seq_len
                ]

                last = logits[final]
                shown = last.masked_fill(~mask, float("-inf"))
                top_ids = shown.topk(TOP_K).indices
                top_ranks = _ranks_of(last.unsqueeze(0), top_ids.unsqueeze(0))[0]
                top15 = [
                    [
                        tokenizer.decode(
                            [int(t)], clean_up_tokenization_spaces=False
                        ),
                        int(r),
                    ]
                    for t, r in zip(top_ids, top_ranks, strict=True)
                ]

                layer_recs.append(
                    {
                        "layer": int(layer),
                        "model": bool(layer == final_layer),
                        "fr": fr,
                        "best": best,
                        "best16": best16,
                        "ent": ent,
                        "top15": top15,
                    }
                )
                del logits

            pred_letter = None
            if fmt == "arc":
                fr_model = layer_recs[-1]["fr"]
                pred_letter = int(np.argmin(fr_model))

            rec = {
                "item_key": item["item_key"],
                "category": item["category"],
                "example_id": item["example_id"],
                "format": fmt,
                "polarity": item["polarity"],
                "condition": item["condition"],
                "label": item["label"],
                "stereo_loc": item["stereo_loc"],
                "counter_loc": item["counter_loc"],
                "unknown_loc": item["unknown_loc"],
                "stereotyped_groups": item["stereotyped_groups"],
                "gold_aligns_stereo": bool(item["label"] == item["stereo_loc"]),
                "seq_len": seq_len,
                "prompt": prompt,
                "tracked": {k: int(v) for k, v in tk.items()},
                "dist_idx": tracked["aux"]["dist_idx"],
                "pred_letter": pred_letter,
                "layers": layer_recs,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            del lens_logits, model_logits
            n_run += 1

        if (idx + 1) % 10 == 0:
            rate = (time.time() - t0) / max(n_run, 1)
            print(
                f"  item {idx + 1}/{len(items)}  {rate:.2f}s/item-format  "
                f"({n_run} new)",
                flush=True,
            )
    fh.close()
    print(f"done -> {records_path}", flush=True)


# --------------------------------------------------------------------------
# fidelity: int8-GPU vs CPU-bf16 rank agreement
# --------------------------------------------------------------------------

def fidelity(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = sample_items(
        Path(args.bbq_dir), args.categories[:3], 2, args.seed
    )[: args.limit or 8]
    layer_subset = list(range(0, 32, 4))

    results = {}
    for tag, device, int8 in [("cpu_bf16", "cpu", False), ("gpu_int8", "cuda", True)]:
        sub_args = argparse.Namespace(**vars(args))
        sub_args.device, sub_args.int8 = device, int8
        model, tokenizer, lens = load_model_and_lens(sub_args)
        runs = []
        for item in items:
            prompt = build_prompts(item)[args.formats[0]]
            tk = tracked_tokens(tokenizer, item)[args.formats[0]]
            lens_logits, model_logits, input_ids = lens.apply(
                model, prompt, positions=[-1], layers=layer_subset
            )
            vec = torch.tensor([tk["stereo"], tk["counter"], tk["unknown"]])
            per_layer = {}
            for layer, logits in lens_logits.items():
                per_layer[layer] = _ranks_of(logits.to(device), vec.to(device))[0].tolist()
            per_layer[model.n_layers - 1] = _ranks_of(
                model_logits.to(device), vec.to(device)
            )[0].tolist()
            runs.append(per_layer)
            del lens_logits, model_logits
        results[tag] = runs
        del model, lens
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    report = {}
    layers = sorted(results["cpu_bf16"][0])
    for layer in layers:
        a, b = [], []
        agree = 0
        for ra, rb in zip(results["cpu_bf16"], results["gpu_int8"], strict=True):
            a.extend(ra[layer])
            b.extend(rb[layer])
            agree += int(np.argmin(ra[layer]) == np.argmin(rb[layer]))
        try:
            from scipy.stats import spearmanr

            rho = float(spearmanr(a, b).statistic)
        except Exception:
            rho = float(
                np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1]
            )
        report[layer] = {
            "spearman": rho,
            "argmin_agree": agree / len(results["cpu_bf16"]),
        }
        print(
            f"layer {layer:>3}: spearman={rho:.4f} "
            f"argmin-agree={agree}/{len(results['cpu_bf16'])}",
            flush=True,
        )
    (out_dir / "fidelity.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {out_dir / 'fidelity.json'}")


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------

def iter_records(out_dir: Path):
    with (out_dir / "records.jsonl").open() as fh:
        for line in fh:
            yield json.loads(line)


def bootstrap_ci(vals: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(rng.choice(vals, size=len(vals), replace=True)))
        for _ in range(n)
    ]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(args) -> None:
    out_dir = Path(args.out)
    recs = list(iter_records(out_dir))
    print(f"{len(recs)} records")
    layers = sorted({lr["layer"] for r in recs for lr in r["layers"]})

    # ---- per-stratum trajectory table ------------------------------------
    rows = []
    for fmt in args.formats:
        for cond in ("ambig", "disambig"):
            for pol in ("neg", "nonneg"):
                for layer in layers:
                    sel = []
                    for r in recs:
                        if (
                            r["format"] == fmt
                            and r["condition"] == cond
                            and r["polarity"] == pol
                        ):
                            lr = next(
                                x for x in r["layers"] if x["layer"] == layer
                            )
                            sel.append((r, lr))
                    if not sel:
                        continue
                    fr_s = np.array([lr["fr"][0] for _, lr in sel], dtype=float)
                    fr_c = np.array([lr["fr"][1] for _, lr in sel], dtype=float)
                    fr_u = np.array([lr["fr"][2] for _, lr in sel], dtype=float)
                    stereo_pref = fr_s < fr_c
                    row = {
                        "format": fmt,
                        "condition": cond,
                        "polarity": pol,
                        "layer": layer,
                        "is_model_row": bool(sel[0][1]["model"]),
                        "n": len(sel),
                        "stereo_prefer": float(np.mean(stereo_pref)),
                        "gap_median": float(np.median(fr_c - fr_s)),
                        "gap_p25": float(np.percentile(fr_c - fr_s, 25)),
                        "gap_p75": float(np.percentile(fr_c - fr_s, 75)),
                        "unknown_prefer": float(
                            np.mean((fr_u < fr_s) & (fr_u < fr_c))
                        ),
                        "median_rank_stereo": float(np.median(fr_s)),
                        "median_rank_counter": float(np.median(fr_c)),
                        "median_rank_unknown": float(np.median(fr_u)),
                    }
                    if fmt == "arc":
                        preds, golds, aligns = [], [], []
                        for r, lr in sel:
                            fr = lr["fr"]
                            order = [r["stereo_loc"], r["counter_loc"], r["unknown_loc"]]
                            pred = order[int(np.argmin(fr))]
                            preds.append(pred)
                            golds.append(r["label"])
                            aligns.append(r["gold_aligns_stereo"])
                        preds, golds, aligns = map(np.array, (preds, golds, aligns))
                        row["accuracy"] = float(np.mean(preds == golds))
                        row["stereo_choice"] = float(
                            np.mean(preds == [r["stereo_loc"] for r, _ in sel])
                        )
                        row["acc_aligned"] = float(
                            np.mean((preds == golds)[aligns])
                        ) if aligns.any() else float("nan")
                        row["acc_conflict"] = float(
                            np.mean((preds == golds)[~aligns])
                        ) if (~aligns).any() else float("nan")
                    rows.append(row)

    traj_path = out_dir / "strata_trajectory.json"
    traj_path.write_text(json.dumps(rows, indent=1))
    cols = list(rows[0])
    with (out_dir / "strata_trajectory.csv").open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for row in rows:
            fh.write(",".join(str(row.get(c, "")) for c in cols) + "\n")

    # ---- per-category table at selected layers ---------------------------
    model_layer = max(layers)
    pick = [layers[0], layers[len(layers) // 2], model_layer]
    cat_rows = []
    for cat in CATEGORIES:
        for fmt in args.formats:
            for cond in ("ambig", "disambig"):
                for layer in pick:
                    sel = [
                        (r, next(x for x in r["layers"] if x["layer"] == layer))
                        for r in recs
                        if r["category"] == cat
                        and r["format"] == fmt
                        and r["condition"] == cond
                    ]
                    if not sel:
                        continue
                    fr_s = np.array([lr["fr"][0] for _, lr in sel], dtype=float)
                    fr_c = np.array([lr["fr"][1] for _, lr in sel], dtype=float)
                    cat_rows.append(
                        {
                            "category": cat,
                            "format": fmt,
                            "condition": cond,
                            "layer": layer,
                            "n": len(sel),
                            "stereo_prefer": float(np.mean(fr_s < fr_c)),
                            "gap_median": float(np.median(fr_c - fr_s)),
                        }
                    )
    cat_path = out_dir / "per_category.json"
    cat_path.write_text(json.dumps(cat_rows, indent=1))

    # ---- bootstrap CIs on the headline ambig stereo-preference -----------
    ci_rows = []
    for fmt in args.formats:
        for layer in pick:
            sel = [
                next(x for x in r["layers"] if x["layer"] == layer)
                for r in recs
                if r["format"] == fmt and r["condition"] == "ambig"
            ]
            if not sel:
                continue
            pref = np.array(
                [lr["fr"][0] < lr["fr"][1] for lr in sel], dtype=float
            )
            lo, hi = bootstrap_ci(pref)
            ci_rows.append(
                {
                    "format": fmt,
                    "layer": layer,
                    "n": len(sel),
                    "stereo_prefer": float(np.mean(pref)),
                    "ci95": [lo, hi],
                }
            )
            print(
                f"ambig pooled {fmt} layer {layer}: "
                f"P(stereo>counter)={np.mean(pref):.3f} CI95=({lo:.3f},{hi:.3f})"
            )
    (out_dir / "bootstrap_ci.json").write_text(json.dumps(ci_rows, indent=1))

    # ---- entity vs final position ----------------------------------------
    ent_rows = []
    for fmt in args.formats:
        for layer in layers:
            e_ranks, f_ranks = [], []
            for r in recs:
                if r["format"] != fmt:
                    continue
                lr = next(x for x in r["layers"] if x["layer"] == layer)
                for ent in lr["ent"]:
                    e_ranks.append(min(ent[1], ent[2]))
                f_ranks.append(min(lr["fr"][0], lr["fr"][1]))
            if e_ranks:
                ent_rows.append(
                    {
                        "format": fmt,
                        "layer": layer,
                        "n_ent": len(e_ranks),
                        "median_best_group_rank_at_mentions": float(np.median(e_ranks)),
                        "median_best_group_rank_at_answer_slot": float(
                            np.median(f_ranks)
                        ),
                    }
                )
    (out_dir / "entity_vs_final.json").write_text(json.dumps(ent_rows, indent=1))

    make_plots(rows, recs, ent_rows, layers, args, out_dir)
    dump_qualitative(recs, pick, out_dir)
    print(f"wrote {traj_path}, {cat_path}, plots, qualitative dump")


def make_plots(rows, recs, ent_rows, layers, args, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def sel(rows, **kw):
        return [
            r
            for r in rows
            if all(r.get(k) == v for k, v in kw.items())
        ]

    # 1. P(stereo preferred) vs layer, 4 strata, one panel per format
    fig, axes = plt.subplots(1, len(args.formats), figsize=(13, 4.5), sharey=True)
    if len(args.formats) == 1:
        axes = [axes]
    for ax, fmt in zip(axes, args.formats, strict=True):
        for cond, pol, style in [
            ("ambig", "neg", "-o"),
            ("ambig", "nonneg", "-s"),
            ("disambig", "neg", "--o"),
            ("disambig", "nonneg", "--s"),
        ]:
            sub = sorted(
                sel(rows, format=fmt, condition=cond, polarity=pol),
                key=lambda r: r["layer"],
            )
            ax.plot(
                [r["layer"] for r in sub],
                [r["stereo_prefer"] for r in sub],
                style,
                ms=3,
                label=f"{cond}/{pol}",
            )
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xlabel("layer")
        ax.set_title(f"P(stereo token outranks counter) - {fmt}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("fraction of items")
    fig.tight_layout()
    fig.savefig(out_dir / "bias_curves.png", dpi=140)
    plt.close(fig)

    # 2. category x layer heatmap of median gap (ambig only), from records
    for fmt in args.formats:
        grid = np.full((len(CATEGORIES), len(layers)), np.nan)
        for ci, cat in enumerate(CATEGORIES):
            cat_sel = [
                r
                for r in recs
                if r["category"] == cat
                and r["format"] == fmt
                and r["condition"] == "ambig"
            ]
            if not cat_sel:
                continue
            for li, layer in enumerate(layers):
                gaps = [
                    lr["fr"][1] - lr["fr"][0]
                    for r in cat_sel
                    for lr in r["layers"]
                    if lr["layer"] == layer
                ]
                if gaps:
                    grid[ci, li] = float(np.median(gaps))
        fig, ax = plt.subplots(figsize=(10, 5.5))
        finite = grid[np.isfinite(grid)]
        if finite.size == 0:
            plt.close(fig)
            continue
        lim = float(np.percentile(np.abs(finite), 95)) or 1.0
        im = ax.imshow(
            grid, aspect="auto", origin="lower", cmap="RdBu_r",
            vmin=-lim, vmax=lim,
            extent=[min(layers) - 0.5, max(layers) + 0.5, -0.5, len(CATEGORIES) - 0.5],
        )
        ax.set_yticks(range(len(CATEGORIES)))
        ax.set_yticklabels(CATEGORIES, fontsize=8)
        ax.set_xlabel("layer")
        ax.set_title(
            f"median rank gap (counter - stereo), ambiguous only - {fmt}\n"
            "blue = stereotype token closer to the top"
        )
        fig.colorbar(im, ax=ax, label="median rank gap")
        fig.tight_layout()
        fig.savefig(out_dir / f"gap_heatmap_{fmt}.png", dpi=140)
        plt.close(fig)

    # 3. ARC accuracy curves, disambig split by gold/stereo alignment
    if "arc" in args.formats:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for key, label, style in [
            ("accuracy", "overall", "-o"),
            ("acc_aligned", "gold aligns w/ stereotype", "-s"),
            ("acc_conflict", "gold conflicts w/ stereotype", "-^"),
        ]:
            sub = sorted(
                [r for r in rows if r["format"] == "arc" and r["condition"] == "disambig" and key in r],
                key=lambda r: r["layer"],
            )
            by_layer = {}
            for r in sub:
                by_layer.setdefault(r["layer"], []).append(r[key])
            ax.plot(
                sorted(by_layer),
                [float(np.nanmean(by_layer[l])) for l in sorted(by_layer)],
                style,
                ms=3,
                label=label,
            )
        ax.set_xlabel("layer")
        ax.set_ylabel("accuracy (letter argmax vs gold)")
        ax.set_title("ARC disambiguated accuracy by depth")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "accuracy_curves_arc.png", dpi=140)
        plt.close(fig)

    # 4. entity-mention vs answer-slot group rank by layer
    fig, axes = plt.subplots(1, len(args.formats), figsize=(13, 4.5), sharey=True)
    if len(args.formats) == 1:
        axes = [axes]
    for ax, fmt in zip(axes, args.formats, strict=True):
        sub = sorted(
            [r for r in ent_rows if r["format"] == fmt], key=lambda r: r["layer"]
        )
        ax.plot(
            [r["layer"] for r in sub],
            [r["median_best_group_rank_at_mentions"] + 1 for r in sub],
            "-o",
            ms=3,
            label="at group-mention positions",
        )
        ax.plot(
            [r["layer"] for r in sub],
            [r["median_best_group_rank_at_answer_slot"] + 1 for r in sub],
            "-s",
            ms=3,
            label="at answer slot",
        )
        ax.set_yscale("log")
        ax.set_xlabel("layer")
        ax.set_title(f"median best group-token rank - {fmt}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "entity_vs_final.png", dpi=140)
    plt.close(fig)
    print("wrote bias_curves.png, gap_heatmap_<fmt>.png, "
          "accuracy_curves_arc.png, entity_vs_final.png")


def dump_qualitative(recs, pick_layers, out_dir: Path, n_items: int = 8) -> None:
    """Top-15 word-like tokens at the answer slot for a few items/layers."""
    seen_cats = set()
    chosen = []
    for r in recs:
        key = (r["category"], r["format"])
        if key not in seen_cats and r["condition"] == "ambig":
            seen_cats.add(key)
            chosen.append(r)
        if len(chosen) >= n_items:
            break
    lines = []
    for r in chosen:
        lines.append(
            f"### {r['item_key']} [{r['format']}] {r['polarity']}/{r['condition']} "
            f"stereotyped_groups={r['stereotyped_groups']}"
        )
        lines.append(f"> {r['prompt'][:400]}")
        lines.append("")
        for lr in r["layers"]:
            if lr["layer"] not in pick_layers:
                continue
            tag = " (model row)" if lr["model"] else ""
            toks = ", ".join(
                f"{t!r}#{rk}" for t, rk in lr["top15"][:TOP_K]
            )
            lines.append(f"- L{lr['layer']}{tag}: {toks}")
        lines.append("")
    path = out_dir / "qualitative_topk.md"
    path.write_text("\n".join(lines))
    print(f"wrote {path}")


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["collect", "summarize", "fidelity"],
                        required=True)
    parser.add_argument("--bbq-dir", default="/home/cataluna84/Workspace/BBQ")
    parser.add_argument("--categories", nargs="+", default=CATEGORIES)
    parser.add_argument("--blocks-per-cat", type=int, default=50)
    parser.add_argument("--formats", nargs="+", default=["arc", "free"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    parser.add_argument("--lens-revision", default="qwen-n1000")
    parser.add_argument(
        "--lens-file",
        default="qwen3.5-4b/jlens/Salesforce-wikitext/"
        "Qwen3.5-4B_jacobian_lens_n1000.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--int8", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--out", default="analysis/out/bbq")
    args = parser.parse_args()

    {"collect": collect, "summarize": summarize, "fidelity": fidelity}[args.mode](args)


if __name__ == "__main__":
    main()
