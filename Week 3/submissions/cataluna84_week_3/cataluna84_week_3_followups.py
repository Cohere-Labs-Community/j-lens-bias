#!/usr/bin/env python3
"""Week 3 follow-up experiments: attacks on the limitations of round one.

Reuses the committed Week 3 harnesses (``week3_winogender`` /
``week3_steering``) for model loading, directions, and passes; both are
imported unmodified.

GPU modes
---------
fullsent   Teacher-forced log-likelihood of all 720 full Winogender
           sentences. The cloze slot's autoregressive probability cannot see
           the disambiguating suffix; the full-sentence LL can, which repairs
           the voided antecedent control (46/60 occupations had identical
           cloze prefixes across antecedent conditions). Also records a
           per-layer Jacobian-lens pronoun read at the sentence-FINAL
           position, which has integrated pronoun + suffix ("persistence"
           of the gender assignment).
domdir     Learned gender direction: per-layer difference of means between
           the 240 male/female minimal-pair full sentences at the final
           position (pairs differ only in the pronoun). Fixes the "pair is
           chosen, not learned" limitation.
domsteer   Dose-response with the learned direction via a norm-preserving
           Householder-style reflection, same protocol as the pair swap.
randseeds  Random-direction arm across 5 fresh seeds (norm-matched), fixing
           the "one random draw" limitation.
gensteer   Vignettes with the swap hook active at EVERY decode step
           (on-the-fly coordinate flip), not just prefill.
bandsweep  Pair swap restricted to 4 narrow sub-bands inside L20-30, to
           localise the causal effect.

CPU mode
--------
summarize  All statistics and figures, including the no-new-GPU reanalyses:
           per-case (he/his/him) BLS correlation, BLS-vs-Bergsma
           discordance, onset-vs-correlation, and the effect/KL frontier.

Run from the repository root, e.g.
    python analysis/week3_followups.py fullsent
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401  (used by summarize)
import numpy as np  # noqa: F401  (used by summarize)
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import week3_steering as st  # noqa: E402
import week3_winogender as wg  # noqa: E402

SUB_BANDS = {
    "L20-22": [20, 21, 22],
    "L23-25": [23, 24, 25],
    "L26-28": [26, 27, 28],
    "L29-30": [29, 30],
}


# --------------------------------------------------------------------------
# shared helpers
# --------------------------------------------------------------------------

def load_sentences(tsv_path: Path) -> dict[tuple, dict]:
    """``{(occupation, participant, label): {gender: row}}`` raw rows."""
    rows = list(csv.DictReader(tsv_path.open(encoding="utf-8"),
                               delimiter="\t"))
    groups: dict[tuple, dict] = {}
    for r in rows:
        groups.setdefault(
            (r["occupation"], r["participant"], r["label"]), {}
        )[r["gender"]] = r
    return groups


def sentence_ll(model, input_ids) -> tuple[float, int]:
    """Total teacher-forced log-likelihood of the sequence."""
    final = model.n_layers - 1
    clean = st.clean_pass(model, input_ids, [final])
    ids = input_ids[0].tolist()
    resid = clean[final]
    total, n = 0.0, 0
    for start in range(0, len(ids) - 1, 32):
        stop = min(start + 32, len(ids) - 1)
        logits = model.unembed(resid[start:stop]).float()
        tgt = torch.tensor(ids[start + 1:stop + 1], device=logits.device)
        total -= float(
            torch.nn.functional.cross_entropy(logits, tgt,
                                              reduction="sum"))
        n += stop - start
    return total, n


def resume_done(records_path: Path, keys: tuple[str, ...]) -> set:
    done = set()
    if records_path.exists():
        for line in records_path.open():
            rec = json.loads(line)
            done.add(tuple(rec[k] for k in keys))
    if done:
        print(f"resuming: {len(done)} rows present", flush=True)
    return done


def occupation_templates(args) -> list[dict]:
    templates = [t for t in wg.load_templates(Path(args.winogender))
                 if t["target_kind"] == "occupation"]
    return templates[: args.limit] if args.limit else templates


def dose_response(model, tokenizer, pids, templates, arms, alphas,
                  records_path: Path, keys: tuple[str, ...]) -> None:
    """Pair-swap dose-response loop shared by randseeds / bandsweep."""
    done = resume_done(records_path, keys)
    final = model.n_layers - 1
    fh = records_path.open("a", encoding="utf-8")
    t0, n_run = time.time(), 0
    for idx, template in enumerate(templates):
        todo = [(a, al) for a in arms for al in alphas
                if (template["tid"], a, al) not in done]
        if not todo:
            continue
        input_ids = model.encode(template["prefix"])
        vec = [pids[f] for f in template["forms"]]
        record_at = sorted({final, *(l for a in arms.values()
                                     for l in a["V"])})
        clean = st.clean_pass(model, input_ids, record_at)
        clean_logits = st.logits_at(model, clean[final], [-1])[0]
        clean_lp = torch.log_softmax(clean_logits.float(), -1)
        clean_vals = [float(clean_lp[i]) for i in vec]
        for arm_name, alpha in todo:
            arm = arms[arm_name]
            c = st.coords(clean, arm["V_pinv"], slice(None))
            steered = st.steered_pass(model, input_ids, arm["V"],
                                      arm["V_pinv"], c, alpha,
                                      slice(None), [final])
            sl = st.logits_at(model, steered[final], [-1])[0]
            lp = torch.log_softmax(sl.float(), -1)
            fh.write(json.dumps({
                "tid": template["tid"], "occupation": template["occupation"],
                "case": template["case"], "forms": template["forms"],
                "arm": arm_name, "alpha": alpha,
                "clean_lp": clean_vals,
                "steered_lp": [float(lp[i]) for i in vec],
                "kl": st.kl_div(sl, clean_logits),
            }) + "\n")
            fh.flush()
            n_run += 1
        if (idx + 1) % 10 == 0 and n_run:
            print(f"  template {idx + 1}/{len(templates)}  "
                  f"{(time.time() - t0) / n_run:.2f}s/pass  ({n_run} new)",
                  flush=True)
    fh.close()
    print(f"done -> {records_path}", flush=True)


def pair_V(hf_model, tokenizer, lens, layers, pair, device):
    W_U = hf_model.get_output_embeddings().weight
    src, tgt = (st.single_token(tokenizer, w) for w in pair)
    return st.token_V(lens, W_U, layers, src, tgt, device)


# --------------------------------------------------------------------------
# fullsent
# --------------------------------------------------------------------------

def fullsent(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "fullsent.jsonl"
    templates = wg.load_templates(Path(args.winogender))
    groups = load_sentences(Path(args.winogender))
    if args.limit:
        templates = templates[: args.limit]
    done = resume_done(records_path, ("tid", "gender"))

    hf_model, model, tokenizer, lens = st.load_all(args)
    del hf_model
    pids = wg.pronoun_ids(tokenizer)
    final = model.n_layers - 1

    fh = records_path.open("a", encoding="utf-8")
    t0, n_run = time.time(), 0
    for idx, template in enumerate(templates):
        key = (template["occupation"], template["participant"],
               str(template["label"]))
        by_gender = groups[key]
        vec = torch.tensor([pids[f] for f in template["forms"]],
                           dtype=torch.long)
        for gender in ("male", "female", "neutral"):
            if (template["tid"], gender) in done:
                continue
            sentence = by_gender[gender]["sentence"]
            input_ids = model.encode(sentence, max_length=args.max_seq_len)
            ll, n_tok = sentence_ll(model, input_ids)
            lens_logits, model_logits, _ = lens.apply(
                model, sentence, positions=[-1],
                max_seq_len=args.max_seq_len, use_jacobian=True)
            layer_recs = []
            for layer, logits_cpu in sorted(dict(lens_logits).items()):
                lp = torch.log_softmax(logits_cpu[0].float(), dim=-1)
                layer_recs.append({
                    "layer": int(layer), "model": False,
                    "lp": [float(lp[int(i)]) for i in vec],
                })
            lp_m = torch.log_softmax(model_logits[0].float(), dim=-1)
            layer_recs.append({
                "layer": final, "model": True,
                "lp": [float(lp_m[int(i)]) for i in vec],
            })
            fh.write(json.dumps({
                "tid": template["tid"], "gender": gender,
                "target_kind": template["target_kind"],
                "case": template["case"],
                "occupation": template["occupation"],
                "ll": ll, "n_tokens": n_tok,
                "layers": layer_recs,
            }) + "\n")
            fh.flush()
            n_run += 1
        if (idx + 1) % 10 == 0 and n_run:
            print(f"  template {idx + 1}/{len(templates)}  "
                  f"{(time.time() - t0) / n_run:.2f}s/unit  ({n_run} new)",
                  flush=True)
    fh.close()
    print(f"done -> {records_path}", flush=True)


# --------------------------------------------------------------------------
# domdir + domsteer
# --------------------------------------------------------------------------

def domdir(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    templates = wg.load_templates(Path(args.winogender))
    groups = load_sentences(Path(args.winogender))
    if args.limit:
        templates = templates[: args.limit]

    hf_model, model, tokenizer, lens = st.load_all(args)
    del hf_model, tokenizer, lens
    layers = list(range(model.n_layers))

    diffs = {l: [] for l in layers}
    t0 = time.time()
    for idx, template in enumerate(templates):
        key = (template["occupation"], template["participant"],
               str(template["label"]))
        by_gender = groups[key]
        hiddens = {}
        for gender in ("male", "female"):
            ids = model.encode(by_gender[gender]["sentence"],
                               max_length=args.max_seq_len)
            resid = st.clean_pass(model, ids, layers)
            hiddens[gender] = {l: resid[l][-1].cpu() for l in layers}
        for l in layers:
            diffs[l].append(hiddens["male"][l] - hiddens["female"][l])
        if (idx + 1) % 40 == 0:
            print(f"  {idx + 1}/{len(templates)} pairs  "
                  f"{(time.time() - t0) / (idx + 1):.2f}s/pair", flush=True)

    out = {}
    for l in layers:
        D = torch.stack(diffs[l])
        d = D.mean(0)
        d_hat = d / d.norm()
        cos = torch.nn.functional.cosine_similarity(
            D, d_hat.unsqueeze(0), dim=-1)
        out[str(l)] = {
            "d": [float(x) for x in d_hat],
            "mean_diff_norm": float(d.norm()),
            "cos_mean": float(cos.mean()), "cos_sd": float(cos.std()),
        }
    for l in layers:
        r = out[str(l)]
        print(f"  L{l:>2}: ||mean diff||={r['mean_diff_norm']:8.2f}  "
              f"pair cos={r['cos_mean']:+.3f}±{r['cos_sd']:.3f}", flush=True)
    (out_dir / "domdir.json").write_text(json.dumps(out))
    print(f"wrote {out_dir / 'domdir.json'}", flush=True)


def _matched_hook(d_hat: torch.Tensor, mag: torch.Tensor, scale: float):
    """Move each position along the learned axis by a *given* norm.

    ``mag`` is the per-position norm of the pair swap's own displacement at
    the same layer, so the two arms differ only in direction, not in size.
    The sign follows the current projection, mirroring the swap's semantics
    (reverse whichever way this item currently leans).
    """
    state = {"done": False}

    def hook(module, inputs, output):
        if state["done"]:
            return output
        state["done"] = True
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        sel = h[0].float()                              # [seq, d]
        sign = torch.sign(sel @ d_hat)                  # [seq]
        step = (scale * mag * sign).unsqueeze(-1) * d_hat
        h[0] = (sel - step).to(h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h

    return hook


def dommatch(args) -> None:
    """Learned axis vs chosen pair at matched displacement norm."""
    out_dir = Path(args.out)
    records_path = out_dir / "dommatch.jsonl"
    dom = json.loads((out_dir / "domdir.json").read_text())
    templates = occupation_templates(args)
    done = resume_done(records_path, ("tid", "alpha"))

    hf_model, model, tokenizer, lens = st.load_all(args)
    pids = wg.pronoun_ids(tokenizer)
    final = model.n_layers - 1
    band = st.BIAS_BAND
    device = model.input_device
    V = pair_V(hf_model, tokenizer, lens, band, args.pair, device)
    V_pinv = st.pinv_of(V)
    d_hats = {l: torch.tensor(dom[str(l)]["d"], device=device) for l in band}

    fh = records_path.open("a", encoding="utf-8")
    t0, n_run = time.time(), 0
    for idx, template in enumerate(templates):
        todo = [a for a in args.alphas if (template["tid"], a) not in done]
        if not todo:
            continue
        input_ids = model.encode(template["prefix"])
        vec = [pids[f] for f in template["forms"]]
        clean = st.clean_pass(model, input_ids, sorted({final, *band}))
        clean_logits = st.logits_at(model, clean[final], [-1])[0]
        clean_lp = torch.log_softmax(clean_logits.float(), -1)
        clean_vals = [float(clean_lp[i]) for i in vec]
        # the pair swap's own displacement norm per position, at alpha = 1
        mags = {}
        for l in band:
            c = clean[l].float() @ V_pinv[l].T
            mags[l] = ((c.flip(dims=[-1]) - c) @ V[l].T).norm(dim=-1)
        for alpha in todo:
            handles = [model.layers[l].register_forward_hook(
                _matched_hook(d_hats[l], mags[l], alpha)) for l in band]
            try:
                steered = st.clean_pass(model, input_ids, [final])
            finally:
                for h in handles:
                    h.remove()
            sl = st.logits_at(model, steered[final], [-1])[0]
            lp = torch.log_softmax(sl.float(), -1)
            fh.write(json.dumps({
                "tid": template["tid"], "occupation": template["occupation"],
                "case": template["case"], "forms": template["forms"],
                "arm": "dom_matched", "alpha": alpha,
                "clean_lp": clean_vals,
                "steered_lp": [float(lp[i]) for i in vec],
                "kl": st.kl_div(sl, clean_logits),
                "mean_step_norm": float(
                    np.mean([float(mags[l].mean()) for l in band]) * alpha),
            }) + "\n")
            fh.flush()
            n_run += 1
        if (idx + 1) % 10 == 0 and n_run:
            print(f"  template {idx + 1}/{len(templates)}  "
                  f"{(time.time() - t0) / n_run:.2f}s/pass  ({n_run} new)",
                  flush=True)
    fh.close()
    print(f"done -> {records_path}", flush=True)


def _reflect_hook(d_hat: torch.Tensor, alpha: float):
    """One-shot Householder-style reflection, scaled by alpha (prefill)."""
    state = {"done": False}

    def hook(module, inputs, output):
        if state["done"]:
            return output
        state["done"] = True
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        sel = h[0].float()                              # [seq, d]
        proj = sel @ d_hat                              # [seq]
        h[0] = (sel - 2 * alpha * proj.unsqueeze(-1) * d_hat).to(h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h

    return hook


def domsteer(args) -> None:
    out_dir = Path(args.out)
    records_path = out_dir / "domsteer.jsonl"
    dom = json.loads((out_dir / "domdir.json").read_text())
    templates = occupation_templates(args)
    done = resume_done(records_path, ("tid", "alpha"))

    hf_model, model, tokenizer, lens = st.load_all(args)
    del hf_model, lens
    pids = wg.pronoun_ids(tokenizer)
    final = model.n_layers - 1
    band = st.BIAS_BAND
    d_hats = {l: torch.tensor(dom[str(l)]["d"], device=model.input_device)
              for l in band}

    fh = records_path.open("a", encoding="utf-8")
    t0, n_run = time.time(), 0
    for idx, template in enumerate(templates):
        todo = [a for a in args.alphas if (template["tid"], a) not in done]
        if not todo:
            continue
        input_ids = model.encode(template["prefix"])
        vec = [pids[f] for f in template["forms"]]
        clean = st.clean_pass(model, input_ids, [final])
        clean_logits = st.logits_at(model, clean[final], [-1])[0]
        clean_lp = torch.log_softmax(clean_logits.float(), -1)
        clean_vals = [float(clean_lp[i]) for i in vec]
        for alpha in todo:
            handles = [model.layers[l].register_forward_hook(
                _reflect_hook(d_hats[l], alpha)) for l in band]
            try:
                steered = st.clean_pass(model, input_ids, [final])
            finally:
                for h in handles:
                    h.remove()
            sl = st.logits_at(model, steered[final], [-1])[0]
            lp = torch.log_softmax(sl.float(), -1)
            fh.write(json.dumps({
                "tid": template["tid"], "occupation": template["occupation"],
                "case": template["case"], "forms": template["forms"],
                "arm": "dom", "alpha": alpha,
                "clean_lp": clean_vals,
                "steered_lp": [float(lp[i]) for i in vec],
                "kl": st.kl_div(sl, clean_logits),
            }) + "\n")
            fh.flush()
            n_run += 1
        if (idx + 1) % 10 == 0 and n_run:
            print(f"  template {idx + 1}/{len(templates)}  "
                  f"{(time.time() - t0) / n_run:.2f}s/pass  ({n_run} new)",
                  flush=True)
    fh.close()
    print(f"done -> {records_path}", flush=True)


# --------------------------------------------------------------------------
# randseeds + bandsweep
# --------------------------------------------------------------------------

def randseeds(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_model, model, tokenizer, lens = st.load_all(args)
    V_main = pair_V(hf_model, tokenizer, lens, st.BIAS_BAND, args.pair,
                    model.input_device)
    del hf_model
    arms = {}
    for seed in range(args.seed + 1, args.seed + 1 + args.n_seeds):
        V = st.random_V(V_main, seed=seed)
        arms[f"random_s{seed}"] = {"V": V, "V_pinv": st.pinv_of(V)}
    pids = wg.pronoun_ids(tokenizer)
    dose_response(model, tokenizer, pids, occupation_templates(args), arms,
                  args.alphas, out_dir / "randseeds.jsonl",
                  ("tid", "arm", "alpha"))


def bandsweep(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_model, model, tokenizer, lens = st.load_all(args)
    arms = {}
    for name, layers in SUB_BANDS.items():
        V = pair_V(hf_model, tokenizer, lens, layers, args.pair,
                   model.input_device)
        arms[f"counter_{name}"] = {"V": V, "V_pinv": st.pinv_of(V)}
        print(f"arm counter_{name}: L{layers[0]}-L{layers[-1]}", flush=True)
    del hf_model
    pids = wg.pronoun_ids(tokenizer)
    dose_response(model, tokenizer, pids, occupation_templates(args), arms,
                  args.alphas, out_dir / "bandsweep.jsonl",
                  ("tid", "arm", "alpha"))


# --------------------------------------------------------------------------
# gensteer: swap active at every decode step
# --------------------------------------------------------------------------

def _swap_hook_dyn(V, V_pinv, alpha):
    """Coordinate flip computed on the fly, applied EVERY forward step.

    On prefill this touches all prompt positions; during cached decoding the
    incoming tensor is the single new position, so generated tokens are
    steered too. The target is the flip of the *current* coordinates (no
    cached clean coordinates exist beyond the prompt).
    """

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        sel = h[0].float()
        c_cur = sel @ V_pinv.T
        delta = (alpha * (c_cur.flip(dims=[-1]) - c_cur)) @ V.T
        h[0] = (sel + delta).to(h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h

    return hook


def gensteer(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_model, model, tokenizer, lens = st.load_all(args)
    device = model.input_device
    sham = st.sham_pair_from(Path(args.steer_out))
    arms = {
        "counter": pair_V(hf_model, tokenizer, lens, st.BIAS_BAND,
                          args.pair, device),
        "sham": pair_V(hf_model, tokenizer, lens, st.BIAS_BAND,
                       sham, device),
    }
    arms = {k: {"V": v, "V_pinv": st.pinv_of(v)} for k, v in arms.items()}

    rows = []
    prompts = st.VIGNETTES[: args.limit] if args.limit else st.VIGNETTES
    for prompt in prompts:
        input_ids = model.encode(prompt, max_length=args.max_seq_len)
        clean_text = tokenizer.decode(
            hf_model.generate(
                input_ids, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )[0][input_ids.shape[1]:], skip_special_tokens=True)
        for arm_name, arm in arms.items():
            for alpha in args.alphas:
                handles = [model.layers[l].register_forward_hook(
                    _swap_hook_dyn(arm["V"][l], arm["V_pinv"][l], alpha))
                    for l in sorted(arm["V"])]
                try:
                    with torch.no_grad():
                        out = hf_model.generate(
                            input_ids, max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
                finally:
                    for h in handles:
                        h.remove()
                rows.append({
                    "prompt": prompt, "arm": arm_name, "alpha": alpha,
                    "clean": clean_text,
                    "steered": tokenizer.decode(
                        out[0][input_ids.shape[1]:],
                        skip_special_tokens=True),
                })
                print(f"[{arm_name} a={alpha}] {prompt[:44]}... -> "
                      f"{rows[-1]['steered'][:70]!r}", flush=True)
    (out_dir / "gensteer.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {out_dir / 'gensteer.json'}", flush=True)


# --------------------------------------------------------------------------
# summarize (CPU only)
# --------------------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


def _spearman(x, y) -> tuple[float, float]:
    from scipy.stats import spearmanr

    r, p = spearmanr(x, y)
    return float(r), float(p)


def _boot_ci(vals, n=4000, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    draws = [float(np.mean(rng.choice(arr, arr.size, replace=True)))
             for _ in range(n)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _pref_layers(rec: dict) -> dict[int, float]:
    """``{layer: lp_female - lp_male}`` from a detection record."""
    return {lr["layer"]: lr["lp"][1] - lr["lp"][0] for lr in rec["layers"]}


def _dose_stats(records: list[dict]) -> dict[tuple, dict]:
    """Mean delta-pref / KL / flip-rate per (arm, alpha)."""
    out: dict[tuple, dict] = {}
    for r in records:
        d = (r["steered_lp"][1] - r["steered_lp"][0]) - (
            r["clean_lp"][1] - r["clean_lp"][0])
        key = (r["arm"], float(r["alpha"]))
        out.setdefault(key, {"d": [], "kl": [], "flip": []})
        out[key]["d"].append(d)
        out[key]["kl"].append(r["kl"])
        out[key]["flip"].append(
            bool(np.sign(r["steered_lp"][1] - r["steered_lp"][0])
                 != np.sign(r["clean_lp"][1] - r["clean_lp"][0])))
    return {k: {"d_pref": float(np.mean(v["d"])),
                "kl": float(np.mean(v["kl"])),
                "flip": float(np.mean(v["flip"])), "n": len(v["d"])}
            for k, v in out.items()}


def summarize(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    det = [r for r in _read_jsonl(Path(args.det_out) / "records.jsonl")
           if r["style"] == "raw" and r["transport"] == "jacobian"]
    det_occ = [r for r in det if r["target_kind"] == "occupation"]
    bls = wg.load_bls(Path(args.bls))
    steer = _read_jsonl(Path(args.steer_out) / "steer_winogender.jsonl")
    summary: dict = {}

    # ---- 1. per-case BLS correlation ----------------------------------
    layers_all = sorted({lr["layer"] for r in det_occ for lr in r["layers"]})
    case_curves: dict[str, list] = {}
    for case in ("he", "his", "him"):
        recs = [r for r in det_occ if r["case"] == case]
        occs = sorted({r["occupation"] for r in recs})
        curve = []
        for layer in layers_all:
            prefs, ys = [], []
            for occ in occs:
                vals = [_pref_layers(r)[layer] for r in recs
                        if r["occupation"] == occ]
                if vals:
                    prefs.append(float(np.mean(vals)))
                    ys.append(bls[occ]["bls"])
            rho, p = _spearman(prefs, ys)
            curve.append({"layer": layer, "rho": rho, "p": p,
                          "n_occ": len(prefs)})
        case_curves[case] = curve
        at29 = next(c for c in curve if c["layer"] == 29)
        note = ("  (low power: case is nearly a property of the occupation "
                "in Winogender)") if at29["n_occ"] < 15 else ""
        print(f"case {case}: n_occ={at29['n_occ']}  "
              f"rho@L29={at29['rho']:+.3f} (p={at29['p']:.2e}){note}",
              flush=True)
    summary["case_split"] = case_curves

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for case, color in (("he", "C0"), ("his", "C1"), ("him", "C2")):
        c = case_curves[case]
        ax.plot([x["layer"] for x in c], [x["rho"] for x in c],
                "-o", ms=3, color=color,
                label=f"{case} (n={c[-1]['n_occ']})")
    ax.axvspan(20, 30, color="orange", alpha=0.08)
    ax.set_xlabel("layer")
    ax.set_ylabel(r"Spearman $\rho$ vs BLS %female")
    ax.set_title("BLS correlation by pronoun case (raw/jacobian)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_case_split.png", dpi=140)
    plt.close(fig)

    # ---- 2. BLS vs Bergsma ground-truth discordance --------------------
    occs = sorted({r["occupation"] for r in det_occ})
    band_pref = {}
    for occ in occs:
        vals = [_pref_layers(r)[l]
                for r in det_occ if r["occupation"] == occ
                for l in range(20, 31)]
        band_pref[occ] = float(np.mean(vals))
    x_bls = np.array([bls[o]["bls"] for o in occs])
    x_berg = np.array([bls[o]["bergsma"] for o in occs])
    y_pref = np.array([band_pref[o] for o in occs])
    rho_bls, p_bls = _spearman(y_pref, x_bls)
    rho_berg, p_berg = _spearman(y_pref, x_berg)
    discordant = sorted(
        (o for o in occs if abs(bls[o]["bls"] - bls[o]["bergsma"]) >= 25),
        key=lambda o: abs(bls[o]["bls"] - bls[o]["bergsma"]), reverse=True)
    sub = {t: _spearman([band_pref[o] for o in discordant],
                        [bls[o][t] for o in discordant])
           for t in ("bls", "bergsma")}
    summary["groundtruth"] = {
        "rho_bls": [rho_bls, p_bls], "rho_bergsma": [rho_berg, p_berg],
        "discordant": {o: {"bls": bls[o]["bls"],
                           "bergsma": bls[o]["bergsma"],
                           "band_pref": band_pref[o]} for o in discordant},
        "discordant_rho": sub,
    }
    print(f"rho(band pref, BLS)={rho_bls:+.3f}  rho(band pref, Bergsma)="
          f"{rho_berg:+.3f}   discordant n={len(discordant)} "
          f"(worst: {', '.join(discordant[:8])})", flush=True)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    sc = a1.scatter(x_bls, x_berg, c=y_pref, cmap="RdBu_r",
                    edgecolor="k", linewidth=0.4)
    for o in discordant[:8]:
        a1.annotate(o, (bls[o]["bls"], bls[o]["bergsma"]),
                    fontsize=6, alpha=0.8)
    a1.plot([0, 100], [0, 100], ":", color="gray", lw=1)
    a1.set_xlabel("BLS %female (2015)")
    a1.set_ylabel("Bergsma %female (web)")
    a1.set_title("two ground truths disagree (colour = lens lean)")
    fig.colorbar(sc, ax=a1, shrink=0.8)
    a2.bar(["all\nBLS", "all\nBergsma", "disc.\nBLS", "disc.\nBergsma"],
           [rho_bls, rho_berg, sub["bls"][0], sub["bergsma"][0]],
           color=["C0", "C1", "C0", "C1"], alpha=0.85)
    a2.set_ylabel(r"Spearman $\rho$ with lens lean")
    a2.set_title("which ground truth does the lens track?")
    a2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_groundtruth.png", dpi=140)
    plt.close(fig)

    # ---- 3. onset of pronoun-mode vs correlation -----------------------
    onset = []
    for layer in layers_all:
        ranks = [[lr["best_pron_rank"] for lr in r["layers"]
                  if lr["layer"] == layer][0] for r in det_occ]
        prefs, ys = [], []
        for occ in occs:
            vals = [_pref_layers(r)[layer] for r in det_occ
                    if r["occupation"] == occ]
            prefs.append(float(np.mean(vals)))
            ys.append(bls[occ]["bls"])
        rho, _ = _spearman(prefs, ys)
        onset.append({"layer": layer,
                      "pron_top1": float(np.mean([r == 1 for r in ranks])),
                      "pron_top5": float(np.mean([r <= 5 for r in ranks])),
                      "rho": rho})
    half5 = next((o["layer"] for o in onset if o["pron_top5"] > 0.5), None)
    rho_curve, _ = _spearman([o["pron_top1"] for o in onset],
                             [o["rho"] for o in onset])
    rho_curve_band, _ = _spearman(
        [o["pron_top1"] for o in onset if 20 <= o["layer"] <= 30],
        [o["rho"] for o in onset if 20 <= o["layer"] <= 30])
    summary["onset"] = {"per_layer": onset,
                        "first_layer_top5_majority": half5,
                        "rho_between_curves": rho_curve,
                        "rho_between_curves_band": rho_curve_band}
    at = {o["layer"]: o for o in onset}
    print(f"pronoun top-5 majority from L{half5}; top-1 fraction "
          f"L20={at[20]['pron_top1']:.2f} L24={at[24]['pron_top1']:.2f} "
          f"L28={at[28]['pron_top1']:.2f} L31={at[31]['pron_top1']:.2f}; "
          f"rho(top1, rho_BLS) all layers {rho_curve:+.3f}, "
          f"band only {rho_curve_band:+.3f}", flush=True)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax2 = ax.twinx()
    ax.bar([o["layer"] for o in onset], [o["pron_top5"] for o in onset],
           color="gray", alpha=0.30, label="pronoun top-5 fraction")
    ax.bar([o["layer"] for o in onset], [o["pron_top1"] for o in onset],
           color="gray", alpha=0.65, label="pronoun top-1 fraction")
    ax2.plot([o["layer"] for o in onset], [o["rho"] for o in onset],
             "-o", ms=3, color="C3", label=r"$\rho$ vs BLS")
    ax.axvspan(20, 30, color="orange", alpha=0.08)
    ax.set_xlabel("layer")
    ax.set_ylabel("pronoun-mode fraction", color="gray")
    ax2.set_ylabel(r"Spearman $\rho$ vs BLS", color="C3")
    ax.set_title("onset: pronoun-mode fraction vs BLS correlation")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_onset.png", dpi=140)
    plt.close(fig)

    _summarize_gpu(args, out_dir, bls, band_pref, steer, summary)

    (out_dir / "followups_summary.json").write_text(
        json.dumps(summary, indent=1, default=str))
    print(f"wrote {out_dir / 'followups_summary.json'}", flush=True)


def _summarize_gpu(args, out_dir: Path, bls, band_pref, steer,
                   summary: dict) -> None:
    band = st.BIAS_BAND

    # ---- 4. full-sentence antecedent contrast + persistence ------------
    fs = _read_jsonl(out_dir / "fullsent.jsonl")
    if fs:
        by_tid: dict[str, dict] = {}
        for r in fs:
            by_tid.setdefault(r["tid"], {})[r["gender"]] = r
        prefs = {}
        for tid, g in by_tid.items():
            if set(g) == {"male", "female", "neutral"}:
                prefs[tid] = {
                    "pref": g["female"]["ll"] - g["male"]["ll"],
                    "target_kind": g["male"]["target_kind"],
                    "occupation": g["male"]["occupation"],
                    "case": g["male"]["case"],
                }
        occ_rows = {t: v for t, v in prefs.items()
                    if v["target_kind"] == "occupation"}
        part_rows = {t: v for t, v in prefs.items()
                     if v["target_kind"] == "participant"}
        occ_pref = {t: v["pref"] for t, v in occ_rows.items()}
        part_pref = {t: v["pref"] for t, v in part_rows.items()}
        pairs = []
        for t, v in occ_rows.items():
            occ, part, _ = t.split(".")
            for t2, v2 in part_rows.items():
                if t2.startswith(f"{occ}.{part}."):
                    pairs.append((v["pref"], v2["pref"]))
        from scipy.stats import wilcoxon

        diffs = np.array([a - b for a, b in pairs])
        wstat, wp = wilcoxon(diffs)
        lo, hi = _boot_ci(diffs)
        occs_fs = sorted({v["occupation"] for v in occ_rows.values()})
        occ_mean = {o: float(np.mean([v["pref"] for v in occ_rows.values()
                                      if v["occupation"] == o]))
                    for o in occs_fs}
        rho_full, p_full = _spearman([occ_mean[o] for o in occs_fs],
                                     [bls[o]["bls"] for o in occs_fs])
        summary["fullsent"] = {
            "n_templates": len(prefs), "n_pairs": len(pairs),
            "mean_pref_occupation": float(np.mean(list(occ_pref.values()))),
            "mean_pref_participant": float(np.mean(list(part_pref.values()))),
            "paired_diff": {"mean": float(diffs.mean()), "ci": [lo, hi],
                            "wilcoxon_p": float(wp)},
            "rho_bls_fullsent": [rho_full, p_full],
        }
        print(f"fullsent: occ {np.mean(list(occ_pref.values())):+.3f} vs "
              f"part {np.mean(list(part_pref.values())):+.3f} nats, paired "
              f"Wilcoxon p={wp:.2e}; rho vs BLS={rho_full:+.3f}", flush=True)

        lay_sorted = sorted({lr["layer"] for r in fs for lr in r["layers"]})
        pers = {"male": [], "female": [], "neutral": []}
        for layer in lay_sorted:
            vals = {"male": [], "female": [], "neutral": []}
            for r in fs:
                lr = next(x for x in r["layers"] if x["layer"] == layer)
                lp = lr["lp"]
                if r["gender"] == "male":
                    vals["male"].append(lp[0] - lp[1])
                elif r["gender"] == "female":
                    vals["female"].append(lp[1] - lp[0])
                else:
                    vals["neutral"].append(lp[2] - max(lp[0], lp[1]))
            for k in vals:
                pers[k].append(float(np.mean(vals[k])))
        summary["persistence"] = {"layers": lay_sorted, **pers}

        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4.0))
        a1.hist(list(occ_pref.values()), bins=30, alpha=0.6,
                label=f"occupation (n={len(occ_pref)})")
        a1.hist(list(part_pref.values()), bins=30, alpha=0.6,
                label=f"participant (n={len(part_pref)})")
        a1.axvline(0, color="gray", ls=":", lw=1)
        a1.set_xlabel("full-sentence pref (nats)")
        a1.set_title("valid antecedent contrast")
        a1.legend(fontsize=8)
        a2.scatter([bls[o]["bls"] for o in occs_fs],
                   [occ_mean[o] for o in occs_fs], s=14, alpha=0.7)
        a2.set_xlabel("BLS %female")
        a2.set_ylabel("mean full-sentence pref")
        a2.set_title(rf"model-level $\rho$={rho_full:+.2f}")
        a2.grid(alpha=0.3)
        for k, c in (("male", "C0"), ("female", "C1"), ("neutral", "C2")):
            a3.plot(lay_sorted, pers[k], "-", ms=2, color=c,
                    label=f"{k} sentence")
        a3.axvspan(20, 30, color="orange", alpha=0.08)
        a3.axhline(0, color="gray", ls=":", lw=1)
        a3.set_xlabel("layer")
        a3.set_ylabel("own-gender pronoun advantage (nats)")
        a3.set_title("persistence at sentence-final position")
        a3.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_fullsent.png", dpi=140)
        plt.close(fig)

    # ---- 5. band sweep heatmap -----------------------------------------
    sweep = _read_jsonl(out_dir / "bandsweep.jsonl")
    stats = _dose_stats(steer + sweep)
    band_rows = [("early L0-11", "counter_early"),
                 *[(k, f"counter_{k}") for k in SUB_BANDS],
                 ("full L20-30", "counter")]
    alphas_show = [0.5, 1.0, 2.0]
    mat = np.full((len(band_rows), len(alphas_show)), np.nan)
    for i, (_, arm) in enumerate(band_rows):
        for j, a in enumerate(alphas_show):
            if (arm, a) in stats:
                mat[i, j] = stats[(arm, a)]["d_pref"]
    summary["band_sweep"] = {
        label: {str(a): (stats[(arm, a)] if (arm, a) in stats else None)
                for a in alphas_show}
        for label, arm in band_rows
    }
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    im = ax.imshow(mat, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(alphas_show)),
                  [rf"$\alpha$={a}" for a in alphas_show])
    ax.set_yticks(range(len(band_rows)), [r[0] for r in band_rows])
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center",
                        va="center", fontsize=8)
    ax.set_title(r"mean $\Delta$pref (nats) by band")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_bands.png", dpi=140)
    plt.close(fig)
    for label, arm in band_rows:
        row = "  ".join(
            f"a={a}: {stats[(arm, a)]['d_pref']:+.2f}"
            f" (flip {stats[(arm, a)]['flip']:.0%})"
            if (arm, a) in stats else f"a={a}: n/a" for a in alphas_show)
        print(f"  band {label:<12} {row}", flush=True)

    # ---- 6. learned direction (domdir + domsteer) ----------------------
    dom_path = out_dir / "domdir.json"
    domsteer = _read_jsonl(out_dir / "domsteer.jsonl")
    if dom_path.exists():
        dom = json.loads(dom_path.read_text())
        layers_d = sorted(int(k) for k in dom)
        cos_band = [dom[str(l)]["cos_mean"] for l in band]
        summary["domdir"] = {
            "band_cos_mean": float(np.mean(cos_band)),
            "band_cos_min": float(np.min(cos_band)),
            "per_layer": {str(l): {"norm": dom[str(l)]["mean_diff_norm"],
                                   "cos": dom[str(l)]["cos_mean"]}
                          for l in layers_d},
        }
        matched = _read_jsonl(out_dir / "dommatch.jsonl")
        dstats = _dose_stats(domsteer + matched)
        summary["domsteer"] = {
            str(a): dstats.get(("dom", a)) for a in
            sorted({r["alpha"] for r in domsteer})}
        if matched:
            summary["dom_matched"] = {
                str(a): dstats.get(("dom_matched", a))
                for a in sorted({r["alpha"] for r in matched})}
            summary["dom_matched_step_norm"] = float(np.mean(
                [r["mean_step_norm"] for r in matched
                 if r["alpha"] == 1.0] or [float("nan")]))
            for a in sorted({r["alpha"] for r in matched}):
                s = dstats[("dom_matched", a)]
                print(f"  dom_matched x{a}: d_pref={s['d_pref']:+.2f} "
                      f"flip={s['flip']:.0%} KL={s['kl']:.3f}", flush=True)
        print(f"domdir band pair-cos mean={np.mean(cos_band):+.3f} "
              f"(min {np.min(cos_band):+.3f})", flush=True)
        for a in sorted({r["alpha"] for r in domsteer}):
            s = dstats[("dom", a)]
            c = stats.get(("counter", a))
            print(f"  dom a={a}: d_pref={s['d_pref']:+.2f} "
                  f"flip={s['flip']:.0%} KL={s['kl']:.3f}"
                  + (f"   | pair: {c['d_pref']:+.2f} KL={c['kl']:.3f}"
                     if c else ""), flush=True)

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0))
        a1.plot(layers_d, [dom[str(l)]["mean_diff_norm"] for l in layers_d],
                "-o", ms=3, label="||mean male-female diff||")
        a1.plot(layers_d, [dom[str(l)]["cos_mean"] for l in layers_d],
                "-s", ms=3, label="pair consistency (cos)")
        a1.axvspan(20, 30, color="orange", alpha=0.08)
        a1.set_xlabel("layer")
        a1.set_title("learned direction quality")
        a1.legend(fontsize=8)
        a1.grid(alpha=0.3)
        for arm, color, mk, lbl in (
                ("dom", "C3", "-o", "learned dir (reflection)"),
                ("dom_matched", "C2", "-s", "learned dir (norm-matched)"),
                ("counter", "C0", "-^", "he/she pair (swap)")):
            src = dstats if arm.startswith("dom") else stats
            xs = sorted(a for (k, a) in src if k == arm)
            if not xs:
                continue
            a2.plot(xs, [src[(arm, a)]["d_pref"] for a in xs], mk, ms=4,
                    color=color, label=lbl)
        a2.set_xlabel(r"$\alpha$")
        a2.set_ylabel(r"mean $\Delta$pref (nats)")
        a2.set_title("learned direction vs chosen pair")
        a2.legend(fontsize=8)
        a2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_dom.png", dpi=140)
        plt.close(fig)

    # ---- 7. seeds null band + effect/KL frontier ------------------------
    seeds = _read_jsonl(out_dir / "randseeds.jsonl")
    if seeds:
        sstats = _dose_stats(seeds)
        summary["randseeds"] = {
            f"s{s}_a{a}": sstats[(f"random_s{s}", a)]
            for (k, a) in sstats for s in [int(k.rsplit("_s", 1)[1])]}
        for a in (1.0, 2.0):
            vals = [sstats[k]["d_pref"] for k in sstats
                    if k[1] == a]
            c = stats.get(("counter", a))
            print(f"  random seeds a={a}: {min(vals):+.2f}..{max(vals):+.2f} "
                  f"vs counter {c['d_pref'] if c else float('nan'):+.2f}",
                  flush=True)

        all_stats = {**stats, **sstats,
                     **_dose_stats(domsteer
                                   + _read_jsonl(out_dir / "dommatch.jsonl"))}
        pts = [(v["kl"], v["d_pref"], k[0], k[1])
               for k, v in all_stats.items() if v["kl"] < 5]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0))
        for a, mk in ((1.0, "o"), (2.0, "s")):
            xs = [sstats[k]["d_pref"] for k in sstats if k[1] == a]
            a1.scatter([a] * len(xs), xs, color="gray", alpha=0.6,
                       marker=mk, label="_")
            c = stats.get(("counter", a))
            if c:
                a1.scatter([a], [c["d_pref"]], color="C3", marker="*",
                           s=120, zorder=5)
        a1.set_xticks([1.0, 2.0], [r"$\alpha$=1", r"$\alpha$=2"])
        a1.set_ylabel(r"mean $\Delta$pref (nats)")
        a1.set_title("random seeds (gray) vs counter (star)")
        a1.grid(alpha=0.3)
        fam = {"counter": "C3", "dom": "C0"}
        for kl, d, arm, _a in pts:
            base = arm.split("_L")[0].split("_s")[0]
            a2.scatter(kl, d, s=18, alpha=0.7,
                       color=fam.get(base, "gray"))
        pts_sorted = sorted(pts)
        best, fx, fy = -np.inf, [], []
        for kl, d, _, _ in pts_sorted:
            if d > best:
                best, fx, fy = d, [*fx, kl], [*fy, d]
        a2.plot(fx, fy, "k--", lw=1, label="Pareto frontier")
        a2.set_xlabel("mean KL (nats)")
        a2.set_ylabel(r"mean $\Delta$pref (nats)")
        a2.set_title("effect vs side-effect, all arms")
        a2.legend(fontsize=8)
        a2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_frontier_seeds.png", dpi=140)
        plt.close(fig)

    # ---- 8. generation-time steering ------------------------------------
    gen_path = out_dir / "gensteer.json"
    if gen_path.exists():
        rows = json.loads(gen_path.read_text())
        masc, fem = ("he", "his", "him"), ("she", "her", "hers")

        def pron_counts(text: str) -> tuple[int, int]:
            words = [w.strip(".,!?\"'()").lower() for w in text.split()]
            return (sum(words.count(p) for p in masc),
                    sum(words.count(p) for p in fem))

        def gender_flip(clean: str, steered: str) -> bool:
            # a masculine-leaning continuation becoming feminine-leaning
            cm, cf = pron_counts(clean)
            sm, sf = pron_counts(steered)
            return (cm > cf and sf > sm) or (cf > cm and sm > sf)

        def degenerate(text: str) -> bool:
            # repetition collapse: few distinct tokens over a long span
            words = text.split()
            return len(words) > 10 and len(set(words)) / len(words) < 0.4

        table = []
        for arm in sorted({r["arm"] for r in rows}):
            for a in sorted({r["alpha"] for r in rows}):
                sel = [r for r in rows if r["arm"] == arm and r["alpha"] == a]
                n_diff = sum(r["clean"] != r["steered"] for r in sel)
                n_pron = sum(pron_counts(r["clean"]) != pron_counts(r["steered"])
                             for r in sel)
                n_flip = sum(gender_flip(r["clean"], r["steered"])
                             for r in sel)
                n_deg = sum(degenerate(r["steered"]) for r in sel)
                table.append({"arm": arm, "alpha": a, "n": len(sel),
                              "changed": n_diff, "pronoun_changed": n_pron,
                              "gender_flipped": n_flip, "degenerate": n_deg})
                print(f"  gensteer {arm} a={a}: {n_diff}/{len(sel)} changed, "
                      f"{n_pron} pronoun counts differ, {n_flip} gender "
                      f"flipped, {n_deg} degenerate", flush=True)
        summary["gensteer"] = table

    _write_findings(out_dir, summary)


def _write_findings(out_dir: Path, summary: dict) -> None:
    """Human-readable delta report: what changed vs the first-round claims."""
    lines = ["# Week 3 follow-up findings", ""]
    cs = summary.get("case_split", {})
    if cs:
        lines += ["## 1. Case split (he / his / him)", ""]
        for case, curve in cs.items():
            at = next(c for c in curve if c["layer"] == 29)
            lines.append(
                f"- `{case}`: rho@L29 = {at['rho']:+.3f} "
                f"(p={at['p']:.1e}, n_occ={at['n_occ']})")
        lines.append("")
    gt = summary.get("groundtruth", {})
    if gt:
        lines += ["## 2. BLS vs Bergsma ground truths", "",
                  f"- rho(lens lean, BLS) = {gt['rho_bls'][0]:+.3f}; "
                  f"rho(lens lean, Bergsma) = {gt['rho_bergsma'][0]:+.3f}",
                  f"- discordant occupations (|diff| >= 25pp): "
                  f"{len(gt['discordant'])}",
                  ""]
    on = summary.get("onset", {})
    if on:
        lines += ["## 3. Onset of pronoun-mode vs signal", "",
                  f"- pronoun top-5 majority from L"
                  f"{on['first_layer_top5_majority']}; correlation between "
                  f"the top-1 fraction curve and the rho curve: "
                  f"{on['rho_between_curves']:+.3f} (all layers), "
                  f"{on['rho_between_curves_band']:+.3f} (band only)", ""]
    fsx = summary.get("fullsent", {})
    if fsx:
        pd = fsx["paired_diff"]
        lines += [
            "## 4. Full-sentence antecedent contrast (repaired control)", "",
            f"- occupation-antecedent pref: {fsx['mean_pref_occupation']:+.3f}"
            f" nats; participant-antecedent: "
            f"{fsx['mean_pref_participant']:+.3f} nats",
            f"- paired diff {pd['mean']:+.3f} "
            f"(CI {pd['ci'][0]:+.3f}..{pd['ci'][1]:+.3f}, "
            f"Wilcoxon p={pd['wilcoxon_p']:.1e}, n={fsx['n_pairs']})",
            f"- model-level rho vs BLS from full sentences: "
            f"{fsx['rho_bls_fullsent'][0]:+.3f} "
            f"(p={fsx['rho_bls_fullsent'][1]:.1e})",
            ""]
    bs = summary.get("band_sweep", {})
    if bs:
        lines += ["## 5. Band sweep (where the swap lives)", ""]
        for label, per_alpha in bs.items():
            cells = "  ".join(
                f"a={a}: {v['d_pref']:+.2f}" if v else f"a={a}: n/a"
                for a, v in per_alpha.items())
            lines.append(f"- {label}: {cells}")
        lines.append("")
    dd = summary.get("domdir", {})
    if dd:
        lines += ["## 6. Learned gender direction", "",
                  f"- band pair-cosine consistency: "
                  f"{dd['band_cos_mean']:+.3f} (min {dd['band_cos_min']:+.3f})",
                  ""]
        for a, v in (summary.get("domsteer") or {}).items():
            if v:
                lines.append(
                    f"- dom a={a}: d_pref={v['d_pref']:+.2f}, "
                    f"flip={v['flip']:.0%}, KL={v['kl']:.3f}")
        lines.append("")
    rs = summary.get("randseeds", {})
    if rs:
        lines += ["## 7. Random-direction seeds", ""]
        for k, v in sorted(rs.items()):
            lines.append(f"- {k}: d_pref={v['d_pref']:+.2f}, "
                         f"KL={v['kl']:.3f}")
        lines.append("")
    gs = summary.get("gensteer", [])
    if gs:
        lines += ["## 8. Generation-time steering", ""]
        for t in gs:
            lines.append(
                f"- {t['arm']} a={t['alpha']}: {t['changed']}/{t['n']} "
                f"changed, {t.get('pronoun_changed', 0)} with different "
                f"pronoun counts, {t.get('gender_flipped', 0)} with a flipped "
                f"pronoun gender, {t.get('degenerate', 0)} degenerate")
        lines.append("")
    (out_dir / "week3_followups.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir / 'week3_followups.md'}", flush=True)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["fullsent", "domdir", "domsteer",
                                     "dommatch", "randseeds", "gensteer",
                                     "bandsweep", "summarize"])
    ap.add_argument("--out", default="analysis/out/week3/followups")
    ap.add_argument("--steer-out", default="analysis/out/week3/steering")
    ap.add_argument("--det-out", default="analysis/out/week3/winogender")
    ap.add_argument("--winogender", default="analysis/data/winogender_test.tsv")
    ap.add_argument("--bls", default="analysis/data/occupations-stats.tsv")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    ap.add_argument("--lens-revision", default="qwen-n1000")
    ap.add_argument("--lens-file",
                    default="qwen3.5-4b/jlens/Salesforce-wikitext/"
                            "Qwen3.5-4B_jacobian_lens_n1000.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--int8", action="store_true", default=True)
    ap.add_argument("--no-int8", dest="int8", action="store_false")
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--pair", default=" he, she")
    ap.add_argument("--alphas", default="0.5,1,2")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    args.pair = [p for p in args.pair.split(",")]
    args.alphas = [float(a) for a in args.alphas.split(",") if a != ""]

    if args.mode == "summarize":
        summarize(args)
        return
    {"fullsent": fullsent, "domdir": domdir, "domsteer": domsteer,
     "dommatch": dommatch, "randseeds": randseeds, "gensteer": gensteer,
     "bandsweep": bandsweep}[args.mode](args)


if __name__ == "__main__":
    main()
