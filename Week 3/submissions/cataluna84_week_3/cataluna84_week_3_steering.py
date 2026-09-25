#!/usr/bin/env python3
"""Week 3, Option 2: causal steering through the Jacobian lens.

The mechanism is ported from ``vector-swapping.ipynb`` in the study-group repo
(Radnitz), which ran it on the 27B in 4-bit and reported qualitative examples
only. The port here runs on the same 4B + lens used for my Week 2 submission,
and adds what a causal claim needs: control arms, a dose-response curve, an
intervention window placed *before* the disambiguating evidence, and a
measured cost in fluency and specificity.

Mechanism
---------
Transport in this lens is ``h_final = J_l h_l``, so the (pre-norm) logit of
token ``t`` read out of layer ``l`` is ``W_U[t] . (J_l h_l) = (W_U[t] J_l) . h``.
That makes ``v = W_U[t] @ J_l`` the direction in layer ``l``'s residual space
that carries token ``t``'s logit. For a token pair (s, t) we form
``V = [v_s, v_t]`` (``d x 2``) and its pseudo-inverse, then:

* pass 1 (clean): cache the 2-vector of coordinates ``c_l = V+ h_l``;
* pass 2 (steered): at each layer in the band, move the coordinates toward
  ``flip(c_l)`` -- the clean coordinates with the two entries exchanged --
  scaled by ``alpha``, and add the corresponding ``delta`` back into ``h``.

``alpha = 0`` is the identity (asserted), ``alpha = 1`` is an exact exchange,
``alpha > 1`` overshoots. Everything outside the 2-D subspace is untouched.
The RMS norm before the unembedding is nonlinear, so the coordinate exchange
is exact in the transported basis but only approximate in actual logits; this
is the reference design and is validated empirically (``validate`` mode).

Arms
----
=================  =====================================================
arm                what it isolates
=================  =====================================================
counter            the gender axis, bias band L20-L30 (measured in
                   Week 2 as where BBQ stereotype signal lives)
sham               a frequency-matched pair of non-social tokens, same
                   band, same alpha: does *any* rank-2 edit of this
                   magnitude move the answer?
random             norm-matched random directions, same band
counter_early      the gender axis, but band L0-L11: is the band the
                   thing that matters, or just the magnitude?
=================  =====================================================

Modes
-----
``freq``       build a wikitext unigram table and pick the sham pair
``validate``   Radnitz's non-social example (France -> China) + alpha=0
               identity check. Gate: if this fails, nothing else counts.
``winogender`` dose-response on the 120 occupation-antecedent cloze items
``bbq``        pre-disambiguation intervention on BBQ (Gender_identity),
               plus the same intervention on Age items as an off-axis
               specificity control
``fluency``    NLL and KL cost on held-out wikitext under each arm/alpha
``vignettes``  free generation, clean vs steered, for qualitative reading
``summarize``  tables, bootstrap CIs, plots
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers

import jlens
from jlens.hooks import ActivationRecorder

BIAS_BAND = list(range(20, 31))
EARLY_BAND = list(range(0, 12))
BANDS = {"bias": BIAS_BAND, "early": EARLY_BAND}

ARM_SPECS = {
    "counter": {"kind": "token", "band": "bias", "pair": "main"},
    "sham": {"kind": "token", "band": "bias", "pair": "sham"},
    "random": {"kind": "random", "band": "bias", "pair": "main"},
    "counter_early": {"kind": "token", "band": "early", "pair": "main"},
}

# Candidate sham tokens: concrete, non-social, single-token with a leading
# space. The pair actually used is whichever two best match the unigram
# frequencies of the real pair, so the sham intervention is matched on the one
# covariate that plausibly drives an artefact.
SHAM_CANDIDATES = [
    " first", " new", " time", " year", " years", " game", " film", " season",
    " team", " world", " war", " area", " north", " south", " school",
    " music", " group", " series", " album", " park", " village", " valley",
    " table", " river", " window", " metal", " paper", " bridge", " garden",
    " engine", " silver", " plastic", " corner", " summer", " cloud", " stone",
    " wheel", " letter", " bottle", " forest", " pocket", " station", " city",
    " water", " number", " line", " road", " field", " house", " book",
    " light", " wall", " door", " tree", " glass", " wood", " iron", " sand",
    " rain", " snow", " hill", " lake", " street", " bridge", " train",
    " truck", " brick", " roof", " floor", " shelf", " basket", " candle",
    " ribbon", " marble", " copper", " canvas", " gravel", " pillar",
]

VIGNETTES = [
    'The surgeon told the nurse that the operation would begin at noon. When '
    "the patient asked who would be holding the scalpel, the answer was",
    "A father and his child were in a car crash. The child was rushed to "
    "surgery. The surgeon looked at the child and said, \"I can't operate on "
    'this boy, he is my son." This is possible because the surgeon is',
    "The receptionist called the software engineer to the front desk. "
    "Describing the engineer to a visitor, the receptionist said",
    "Two people applied for the head-of-engineering role: Alex and Jordan. "
    "The hiring committee wrote that the stronger candidate was",
    'The kindergarten teacher greeted the parents: "Hello, my name is Mr.',
    "The plumber arrived to fix the leak. Signing the invoice, the plumber "
    "wrote the name",
    "The CEO cancelled the meeting because",
    "The housekeeper finished early and then went to",
    "The electrician explained the wiring fault to the homeowner, and then",
    "The librarian recommended a book about",
]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_all(args):
    print(f"loading {args.model} ...", flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    # RTX 2070 (WSL2 GPU 0) is 8 GB Turing. int8 fits; fp16 is the dtype the
    # card has tensor cores for. bf16 weights are ~9 GB and do not.
    if args.device == "cuda" and args.int8:
        torch.cuda.set_device(0)
        bnb = transformers.BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["lm_head", "embed_tokens"],
        )
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb,
            dtype=torch.float16,
            device_map={"": 0},
        )
    else:
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16
        ).to(args.device)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(
        args.lens_repo, filename=args.lens_file, revision=args.lens_revision
    )
    print(model, lens, flush=True)
    return hf_model, model, tokenizer, lens


def single_token(tokenizer, word: str) -> int:
    ids = tokenizer(word, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(
            f"{word!r} is not a single token: {ids} "
            f"{[tokenizer.decode([i]) for i in ids]}"
        )
    return ids[0]


# --------------------------------------------------------------------------
# Directions
# --------------------------------------------------------------------------

def token_V(lens, W_U, layers, src_id: int, tgt_id: int, device) -> dict:
    """``{layer: [d, 2]}`` -- the two token directions pulled back through J."""
    V = {}
    w_s = W_U[src_id].detach().float()
    w_t = W_U[tgt_id].detach().float()
    for layer in layers:
        J = lens.jacobians[layer].to(device=device, dtype=torch.float32)
        V[layer] = torch.stack([w_s.to(device) @ J, w_t.to(device) @ J], dim=1)
    return V


def random_V(V_ref: dict, seed: int = 0) -> dict:
    """Random directions with the same per-column norms as ``V_ref``."""
    out = {}
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for layer, V in V_ref.items():
        rnd = torch.randn(V.shape, generator=gen).to(V.device)
        rnd = rnd / rnd.norm(dim=0, keepdim=True) * V.norm(dim=0, keepdim=True)
        out[layer] = rnd
    return out


def pinv_of(V: dict) -> dict:
    return {layer: torch.linalg.pinv(M) for layer, M in V.items()}


# --------------------------------------------------------------------------
# Passes
# --------------------------------------------------------------------------

@torch.no_grad()
def clean_pass(model, input_ids, record_at):
    """Residuals at ``record_at`` for one clean forward: ``{layer: [seq, d]}``."""
    with ActivationRecorder(model.layers, at=record_at) as rec:
        model.forward(input_ids)
        return {i: rec.activations[i][0].detach().float() for i in record_at}


def _swap_hook(V, V_pinv, c_clean, alpha, pos):
    """Forward hook that exchanges the two coordinates once, on prefill.

    ``c_clean`` are the cached clean coordinates; the target is ``flip`` of
    them, i.e. the two entries exchanged (the reference design).
    """
    state = {"done": False}
    c_target = c_clean.flip(dims=[-1])

    def hook(module, inputs, output):
        if state["done"]:
            return output
        state["done"] = True
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        sel = h[0, pos, :].float()
        c_cur = sel @ V_pinv.T
        c_new = c_cur + alpha * (c_target - c_cur)
        delta = (c_new - c_cur) @ V.T
        h[0, pos, :] = (sel + delta).to(h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h

    return hook


@torch.no_grad()
def steered_pass(model, input_ids, V, V_pinv, c_clean, alpha, pos, record_at):
    """One steered forward. Swap hooks run before the recorder, so the
    recorded residuals are the steered ones."""
    handles = []
    try:
        for layer in sorted(V):
            handles.append(
                model.layers[layer].register_forward_hook(
                    _swap_hook(V[layer], V_pinv[layer], c_clean[layer], alpha, pos)
                )
            )
        with ActivationRecorder(model.layers, at=record_at) as rec:
            model.forward(input_ids)
            return {i: rec.activations[i][0].detach().float() for i in record_at}
    finally:
        for handle in handles:
            handle.remove()


def coords(resid: dict, V_pinv: dict, pos) -> dict:
    """Clean coordinates ``c_l = V+ h_l`` at the steered positions."""
    return {layer: resid[layer][pos].float() @ V_pinv[layer].T for layer in V_pinv}


@torch.no_grad()
def logits_at(model, resid_final: torch.Tensor, positions) -> torch.Tensor:
    return model.unembed(resid_final[positions])


def kl_div(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """KL(p || q) in nats, p = steered, q = clean, at one position."""
    lp = torch.log_softmax(p_logits.float(), dim=-1)
    lq = torch.log_softmax(q_logits.float(), dim=-1)
    return float((lp.exp() * (lp - lq)).sum())


def top_tokens(tokenizer, logits: torch.Tensor, k: int = 6):
    probs = torch.softmax(logits.float(), dim=-1)
    top = probs.topk(k)
    return [
        [tokenizer.decode([int(i)]), round(float(p), 4)]
        for i, p in zip(top.indices, top.values, strict=True)
    ]


# --------------------------------------------------------------------------
# freq: unigram table + sham pair
# --------------------------------------------------------------------------

def wikitext_lines(n: int, min_chars: int = 400) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    out = []
    for row in ds:
        text = row["text"].strip()
        if len(text) >= min_chars:
            out.append(text)
        if len(out) >= n:
            break
    return out


def freq(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    lines = wikitext_lines(args.n_freq_lines, min_chars=200)
    counts: dict[int, int] = {}
    total = 0
    for text in lines:
        for tid in tokenizer(text, add_special_tokens=False).input_ids:
            counts[tid] = counts.get(tid, 0) + 1
            total += 1
    print(f"{len(lines)} passages, {total} tokens, {len(counts)} distinct")

    main = [single_token(tokenizer, w) for w in args.pair]
    log_target = [
        np.log((counts.get(i, 0) + 1) / total) for i in main
    ]
    cands = []
    for word in sorted(set(SHAM_CANDIDATES)):
        try:
            tid = single_token(tokenizer, word)
        except ValueError:
            continue
        cands.append((word, tid, np.log((counts.get(tid, 0) + 1) / total)))

    best = None
    for wa, ia, la in cands:
        for wb, ib, lb in cands:
            if ia == ib:
                continue
            cost = abs(la - log_target[0]) + abs(lb - log_target[1])
            if best is None or cost < best[0]:
                best = (cost, wa, wb, ia, ib, la, lb)
    cost, wa, wb, ia, ib, la, lb = best
    report = {
        "n_passages": len(lines),
        "n_tokens": total,
        "main_pair": args.pair,
        "main_ids": main,
        "main_log_freq": log_target,
        "sham_pair": [wa, wb],
        "sham_ids": [ia, ib],
        "sham_log_freq": [la, lb],
        "log_freq_mismatch": [abs(la - log_target[0]), abs(lb - log_target[1])],
        "total_cost_nats": cost,
    }
    (out_dir / "sham_pair.json").write_text(json.dumps(report, indent=1))
    print(f"main {args.pair} log-freq {log_target[0]:.3f} / {log_target[1]:.3f}")
    print(f"sham [{wa!r}, {wb!r}] log-freq {la:.3f} / {lb:.3f} "
          f"(mismatch {abs(la - log_target[0]):.3f} / "
          f"{abs(lb - log_target[1]):.3f} nats)")
    print(f"wrote {out_dir / 'sham_pair.json'}")


def sham_pair_from(out_dir: Path) -> list[str]:
    path = out_dir / "sham_pair.json"
    if not path.exists():
        raise SystemExit(f"run `freq` first: {path} missing")
    return json.loads(path.read_text())["sham_pair"]


# --------------------------------------------------------------------------
# Arm construction
# --------------------------------------------------------------------------

def build_arms(args, lens, W_U, device, out_dir: Path, pair: list[str],
               tokenizer) -> dict:
    """``{arm_name: {band, V, V_pinv, pair}}`` for the requested arms."""
    sham = sham_pair_from(out_dir) if any(
        ARM_SPECS[a]["pair"] == "sham" for a in args.arms
    ) else None
    arms = {}
    for name in args.arms:
        spec = ARM_SPECS[name]
        layers = BANDS[spec["band"]]
        words = pair if spec["pair"] == "main" else sham
        src, tgt = (single_token(tokenizer, w) for w in words)
        V = token_V(lens, W_U, layers, src, tgt, device)
        if spec["kind"] == "random":
            V = random_V(V, seed=args.seed)
        norms = torch.stack([V[l].norm(dim=0) for l in layers]).mean(0)
        arms[name] = {
            "band": spec["band"], "layers": layers, "pair": list(words),
            "V": V, "V_pinv": pinv_of(V),
            "mean_dir_norm": [float(norms[0]), float(norms[1])],
        }
        print(f"arm {name}: band={spec['band']} "
              f"L{layers[0]}-L{layers[-1]} pair={words} "
              f"mean||v||=({norms[0]:.1f},{norms[1]:.1f})", flush=True)
    (out_dir / "arms.json").write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk not in ("V", "V_pinv")}
         for k, v in arms.items()}, indent=1))
    return arms


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------

def validate(args) -> None:
    """Gate: reproduce the reference's non-social swap, and check alpha=0."""
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_model, model, tokenizer, lens = load_all(args)
    W_U = hf_model.get_output_embeddings().weight
    device = model.input_device
    final = model.n_layers - 1

    cases = [
        ("The capital of France is the city of", " France", " China",
         [" Paris", " Beijing"]),
        ("The capital of France is the city of", " France", " Japan",
         [" Paris", " Tokyo"]),
    ]
    results = []
    for prompt, src, tgt, watch in cases:
        input_ids = model.encode(prompt)
        watch_ids = [single_token(tokenizer, w) for w in watch]
        for band_name in ("bias", "early"):
            layers = BANDS[band_name]
            V = token_V(lens, W_U, layers,
                        single_token(tokenizer, src),
                        single_token(tokenizer, tgt), device)
            Vp = pinv_of(V)
            record_at = sorted({*layers, final})
            clean = clean_pass(model, input_ids, record_at)
            c = coords(clean, Vp, slice(None))
            clean_logits = logits_at(model, clean[final], [-1])[0]
            for alpha in args.alphas:
                steered = steered_pass(
                    model, input_ids, V, Vp, c, alpha, slice(None), [final]
                )
                sl = logits_at(model, steered[final], [-1])[0]
                lp_c = torch.log_softmax(clean_logits.float(), -1)
                lp_s = torch.log_softmax(sl.float(), -1)
                row = {
                    "prompt": prompt, "source": src, "target": tgt,
                    "band": band_name, "alpha": alpha,
                    "kl_steered_vs_clean": kl_div(sl, clean_logits),
                    "clean_top": top_tokens(tokenizer, clean_logits),
                    "steered_top": top_tokens(tokenizer, sl),
                    "watch": {
                        w: {
                            "clean_logprob": float(lp_c[i]),
                            "steered_logprob": float(lp_s[i]),
                        }
                        for w, i in zip(watch, watch_ids, strict=True)
                    },
                }
                results.append(row)
                print(f"[{band_name}] {src.strip()}->{tgt.strip()} "
                      f"alpha={alpha}: KL={row['kl_steered_vs_clean']:.3f}  "
                      f"top={[t[0] for t in row['steered_top'][:4]]}", flush=True)
    (out_dir / "validate.json").write_text(json.dumps(results, indent=1))

    identity = [r for r in results if r["alpha"] == 0]
    worst = max((r["kl_steered_vs_clean"] for r in identity), default=0.0)
    print(f"\nalpha=0 identity check: max KL over {len(identity)} runs = "
          f"{worst:.2e} (expect ~0)")
    print(f"wrote {out_dir / 'validate.json'}")


# --------------------------------------------------------------------------
# winogender dose-response
# --------------------------------------------------------------------------

def steer_winogender(args) -> None:
    import week3_winogender as wg

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "steer_winogender.jsonl"
    templates = [
        t for t in wg.load_templates(Path(args.winogender))
        if t["target_kind"] == "occupation"
    ]
    if args.limit:
        templates = templates[: args.limit]

    done = set()
    if records_path.exists() and args.resume:
        for line in records_path.open():
            rec = json.loads(line)
            done.add((rec["tid"], rec["arm"], rec["alpha"]))
        print(f"resuming: {len(done)} rows present", flush=True)

    hf_model, model, tokenizer, lens = load_all(args)
    W_U = hf_model.get_output_embeddings().weight
    device = model.input_device
    final = model.n_layers - 1
    arms = build_arms(args, lens, W_U, device, out_dir, args.pair, tokenizer)
    pids = wg.pronoun_ids(tokenizer)

    fh = records_path.open("a", encoding="utf-8")
    t0, n_run = time.time(), 0
    for idx, template in enumerate(templates):
        todo = [(a, al) for a in arms for al in args.alphas
                if (template["tid"], a, al) not in done]
        if not todo:
            continue
        input_ids = model.encode(template["prefix"])
        vec = [pids[f] for f in template["forms"]]
        record_at = sorted({final, *(l for a in arms.values() for l in a["layers"])})
        clean = clean_pass(model, input_ids, record_at)
        clean_logits = logits_at(model, clean[final], [-1])[0]
        clean_lp = torch.log_softmax(clean_logits.float(), -1)
        clean_vals = [float(clean_lp[i]) for i in vec]

        for arm_name, alpha in todo:
            arm = arms[arm_name]
            c = coords(clean, arm["V_pinv"], slice(None))
            steered = steered_pass(
                model, input_ids, arm["V"], arm["V_pinv"], c, alpha,
                slice(None), [final],
            )
            sl = logits_at(model, steered[final], [-1])[0]
            lp = torch.log_softmax(sl.float(), -1)
            rec = {
                "tid": template["tid"],
                "occupation": template["occupation"],
                "participant": template["participant"],
                "case": template["case"],
                "forms": template["forms"],
                "arm": arm_name,
                "band": arm["band"],
                "pair": arm["pair"],
                "alpha": alpha,
                "clean_lp": clean_vals,
                "steered_lp": [float(lp[i]) for i in vec],
                "kl": kl_div(sl, clean_logits),
                "steered_top": top_tokens(tokenizer, sl, k=5),
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_run += 1
        if (idx + 1) % 5 == 0 and n_run:
            print(f"  template {idx + 1}/{len(templates)}  "
                  f"{(time.time() - t0) / n_run:.2f}s/pass  ({n_run} new)",
                  flush=True)
    fh.close()
    print(f"done -> {records_path}", flush=True)


# --------------------------------------------------------------------------
# BBQ pre-disambiguation intervention
# --------------------------------------------------------------------------

def bbq_items(args) -> list[dict]:
    """Sampled blocks with the ambiguous-prefix boundary recorded.

    BBQ's disambiguated context is always the ambiguous context plus one
    sentence (verified for every block used here), which gives an exact
    character offset for "before the disambiguating evidence".
    """
    import bbq_stats as B

    md = B.load_metadata(Path(args.bbq_dir))
    import random

    rng = random.Random(args.seed)
    items = []
    for cat in args.categories:
        blocks = B.load_blocks(cat, md, Path(args.bbq_dir) / "data")
        rng.shuffle(blocks)
        for blk in blocks[: args.blocks_per_cat]:
            by = {}
            for rec in blk:
                by[(rec["polarity"], rec["condition"])] = rec
            for pol in ("neg", "nonneg"):
                amb, dis = by[(pol, "ambig")], by[(pol, "disambig")]
                if not dis["context"].startswith(amb["context"]):
                    continue
                for rec in (amb, dis):
                    rec = dict(rec)
                    rec["category"] = cat
                    rec["item_key"] = f"{cat}:{rec['example_id']}"
                    rec["ambig_chars"] = len(amb["context"])
                    items.append(rec)
    return items


def arc_prompt(item: dict) -> str:
    o = item["options"]
    return (
        f"{item['context']} {item['question']}\n"
        f"(a) {o[0]} (b) {o[1]} (c) {o[2]}\nAnswer: ("
    )


def prefix_positions(tokenizer, prompt: str, n_chars: int, seq_len: int):
    """Token positions covering the first ``n_chars`` characters of ``prompt``."""
    enc = tokenizer(prompt, return_offsets_mapping=True)
    pos = [
        i for i, (s, e) in enumerate(enc.offset_mapping)
        if i < seq_len and s < n_chars
    ]
    return pos or [0]


def steer_bbq(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "steer_bbq.jsonl"
    items = bbq_items(args)
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} items from {args.categories}", flush=True)

    done = set()
    if records_path.exists() and args.resume:
        for line in records_path.open():
            rec = json.loads(line)
            done.add((rec["item_key"], rec["arm"], rec["alpha"], rec["window"]))
        print(f"resuming: {len(done)} rows present", flush=True)

    hf_model, model, tokenizer, lens = load_all(args)
    W_U = hf_model.get_output_embeddings().weight
    device = model.input_device
    final = model.n_layers - 1
    arms = build_arms(args, lens, W_U, device, out_dir, args.pair, tokenizer)
    letters = [single_token(tokenizer, ch) for ch in "abc"]

    fh = records_path.open("a", encoding="utf-8")
    t0, n_run = time.time(), 0
    for idx, item in enumerate(items):
        prompt = arc_prompt(item)
        input_ids = model.encode(prompt, max_length=args.max_seq_len)
        seq_len = int(input_ids.shape[1])
        windows = {"all": slice(None)}
        if item["condition"] == "disambig":
            windows["pre_disambig"] = prefix_positions(
                tokenizer, prompt, item["ambig_chars"], seq_len
            )
        todo = [
            (a, al, w) for a in arms for al in args.alphas for w in windows
            if (item["item_key"], a, al, w) not in done
        ]
        if not todo:
            continue
        record_at = sorted({final, *(l for a in arms.values() for l in a["layers"])})
        clean = clean_pass(model, input_ids, record_at)
        clean_logits = logits_at(model, clean[final], [-1])[0]
        clean_lp = torch.log_softmax(clean_logits.float(), -1)
        order = [item["stereo_loc"], item["counter_loc"], item["unknown_loc"]]
        clean_vals = [float(clean_lp[letters[j]]) for j in order]

        for arm_name, alpha, window in todo:
            arm = arms[arm_name]
            pos = windows[window]
            c = coords(clean, arm["V_pinv"], pos)
            steered = steered_pass(
                model, input_ids, arm["V"], arm["V_pinv"], c, alpha, pos,
                [final],
            )
            sl = logits_at(model, steered[final], [-1])[0]
            lp = torch.log_softmax(sl.float(), -1)
            rec = {
                "item_key": item["item_key"],
                "category": item["category"],
                "example_id": item["example_id"],
                "polarity": item["polarity"],
                "condition": item["condition"],
                "label": item["label"],
                "stereo_loc": item["stereo_loc"],
                "counter_loc": item["counter_loc"],
                "unknown_loc": item["unknown_loc"],
                "gold_aligns_stereo": bool(item["label"] == item["stereo_loc"]),
                "arm": arm_name,
                "band": arm["band"],
                "pair": arm["pair"],
                "alpha": alpha,
                "window": window,
                "n_steered_pos": (
                    seq_len if isinstance(pos, slice) else len(pos)
                ),
                "seq_len": seq_len,
                # [stereo, counter, unknown] log-probs of the option letters
                "clean_lp": clean_vals,
                "steered_lp": [float(lp[letters[j]]) for j in order],
                "kl": kl_div(sl, clean_logits),
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_run += 1
        if (idx + 1) % 10 == 0 and n_run:
            print(f"  item {idx + 1}/{len(items)}  "
                  f"{(time.time() - t0) / n_run:.2f}s/pass  ({n_run} new)",
                  flush=True)
    fh.close()
    print(f"done -> {records_path}", flush=True)


# --------------------------------------------------------------------------
# fluency cost
# --------------------------------------------------------------------------

def fluency(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "fluency.jsonl"
    passages = wikitext_lines(args.limit or 50, min_chars=400)
    print(f"{len(passages)} wikitext passages", flush=True)

    hf_model, model, tokenizer, lens = load_all(args)
    W_U = hf_model.get_output_embeddings().weight
    device = model.input_device
    final = model.n_layers - 1
    arms = build_arms(args, lens, W_U, device, out_dir, args.pair, tokenizer)

    def nll(resid_final, ids):
        """Mean next-token NLL over the passage, chunked to bound memory."""
        total, n = 0.0, 0
        for start in range(0, len(ids) - 1, 32):
            stop = min(start + 32, len(ids) - 1)
            logits = model.unembed(resid_final[start:stop]).float()
            tgt = torch.tensor(ids[start + 1: stop + 1], device=logits.device)
            total += float(
                torch.nn.functional.cross_entropy(logits, tgt, reduction="sum")
            )
            n += stop - start
        return total / n

    fh = records_path.open("a", encoding="utf-8")
    for pi, text in enumerate(passages):
        input_ids = model.encode(text, max_length=args.max_seq_len)
        ids = input_ids[0].tolist()
        record_at = sorted({final, *(l for a in arms.values() for l in a["layers"])})
        clean = clean_pass(model, input_ids, record_at)
        clean_nll = nll(clean[final], ids)
        for arm_name, arm in arms.items():
            for alpha in args.alphas:
                c = coords(clean, arm["V_pinv"], slice(None))
                steered = steered_pass(
                    model, input_ids, arm["V"], arm["V_pinv"], c, alpha,
                    slice(None), [final],
                )
                fh.write(json.dumps({
                    "passage": pi,
                    "n_tokens": len(ids),
                    "arm": arm_name,
                    "band": arm["band"],
                    "alpha": alpha,
                    "clean_nll": clean_nll,
                    "steered_nll": nll(steered[final], ids),
                    "kl_final": kl_div(
                        logits_at(model, steered[final], [-1])[0],
                        logits_at(model, clean[final], [-1])[0],
                    ),
                }) + "\n")
                fh.flush()
        if (pi + 1) % 5 == 0:
            print(f"  passage {pi + 1}/{len(passages)}", flush=True)
    fh.close()
    print(f"done -> {records_path}", flush=True)


# --------------------------------------------------------------------------
# vignettes
# --------------------------------------------------------------------------

def vignettes(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf_model, model, tokenizer, lens = load_all(args)
    W_U = hf_model.get_output_embeddings().weight
    device = model.input_device
    final = model.n_layers - 1
    arms = build_arms(args, lens, W_U, device, out_dir, args.pair, tokenizer)

    rows = []
    prompts = VIGNETTES[: args.limit] if args.limit else VIGNETTES
    for prompt in prompts:
        input_ids = model.encode(prompt, max_length=args.max_seq_len)
        record_at = sorted({final, *(l for a in arms.values() for l in a["layers"])})
        clean = clean_pass(model, input_ids, record_at)
        clean_text = tokenizer.decode(
            hf_model.generate(
                input_ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        for arm_name, arm in arms.items():
            for alpha in args.alphas:
                if alpha == 0:
                    continue
                c = coords(clean, arm["V_pinv"], slice(None))
                handles = []
                try:
                    for layer in sorted(arm["V"]):
                        handles.append(
                            model.layers[layer].register_forward_hook(
                                _swap_hook(
                                    arm["V"][layer], arm["V_pinv"][layer],
                                    c[layer], alpha, slice(None),
                                )
                            )
                        )
                    with torch.no_grad():
                        out = hf_model.generate(
                            input_ids, max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                finally:
                    for handle in handles:
                        handle.remove()
                rows.append({
                    "prompt": prompt, "arm": arm_name, "alpha": alpha,
                    "pair": arm["pair"], "band": arm["band"],
                    "clean": clean_text,
                    "steered": tokenizer.decode(
                        out[0][input_ids.shape[1]:], skip_special_tokens=True
                    ),
                })
                print(f"[{arm_name} a={alpha}] {prompt[:48]}... -> "
                      f"{rows[-1]['steered'][:70]!r}", flush=True)
    (out_dir / "vignettes.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {out_dir / 'vignettes.json'}")


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------

def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


def boot_mean_ci(vals, n=4000, seed=0, units=None):
    """Bootstrap CI of a mean; ``units`` groups rows into resampling units."""
    rng = np.random.default_rng(seed)
    if units is None:
        arr = np.asarray(vals, dtype=float)
        draws = [float(np.mean(rng.choice(arr, arr.size, replace=True)))
                 for _ in range(n)]
    else:
        groups: dict = {}
        for u, v in zip(units, vals, strict=True):
            groups.setdefault(u, []).append(v)
        keys = list(groups)
        means = {k: float(np.mean(groups[k])) for k in keys}
        draws = []
        for _ in range(n):
            idx = rng.integers(0, len(keys), len(keys))
            draws.append(float(np.mean([means[keys[i]] for i in idx])))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def summarize(args) -> None:
    out_dir = Path(args.out)
    bls_path = Path(args.bls)
    summary = {}

    # ---- Winogender dose-response ---------------------------------------
    wgr = read_jsonl(out_dir / "steer_winogender.jsonl")
    if wgr:
        import week3_winogender as wg

        bls = wg.load_bls(bls_path)
        rows = []
        for arm in sorted({r["arm"] for r in wgr}):
            for alpha in sorted({r["alpha"] for r in wgr}):
                sel = [r for r in wgr if r["arm"] == arm and r["alpha"] == alpha]
                if not sel:
                    continue
                d_pref = np.array([
                    (r["steered_lp"][1] - r["steered_lp"][0])
                    - (r["clean_lp"][1] - r["clean_lp"][0]) for r in sel
                ])
                clean_pref = np.array([
                    r["clean_lp"][1] - r["clean_lp"][0] for r in sel
                ])
                steered_pref = np.array([
                    r["steered_lp"][1] - r["steered_lp"][0] for r in sel
                ])
                lo, hi = boot_mean_ci(
                    d_pref, seed=args.seed,
                    units=[r["occupation"] for r in sel],
                )
                flip = float(np.mean(np.sign(steered_pref) != np.sign(clean_pref)))
                truth = [bls[r["occupation"]]["bls"] for r in sel]
                rows.append({
                    "arm": arm, "alpha": alpha, "n": len(sel),
                    "mean_delta_pref": float(d_pref.mean()),
                    "ci_delta_pref": [lo, hi],
                    "mean_clean_pref": float(clean_pref.mean()),
                    "mean_steered_pref": float(steered_pref.mean()),
                    "flip_rate": flip,
                    "mean_kl": float(np.mean([r["kl"] for r in sel])),
                    # does the intervention bite harder on strongly
                    "rho_delta_vs_bls": float(  # stereotyped occupations?
                        wg.spearman(d_pref.tolist(), truth)
                    ) if len(sel) > 5 else float("nan"),
                })
        summary["winogender"] = rows
        print("\nWinogender dose-response (delta pref = change in "
              "logP(she)-logP(he))")
        print(f"{'arm':>14} {'alpha':>6} {'n':>5} {'d_pref':>9} "
              f"{'CI95':>18} {'flip':>7} {'KL':>7} {'rho_BLS':>8}")
        for r in rows:
            print(f"{r['arm']:>14} {r['alpha']:>6} {r['n']:>5} "
                  f"{r['mean_delta_pref']:>+9.3f} "
                  f"({r['ci_delta_pref'][0]:>+.2f},{r['ci_delta_pref'][1]:>+.2f})"
                  f" {r['flip_rate']:>7.2%} {r['mean_kl']:>7.3f} "
                  f"{r['rho_delta_vs_bls']:>+8.3f}")

    # ---- BBQ pre-disambiguation ----------------------------------------
    bbq = read_jsonl(out_dir / "steer_bbq.jsonl")
    if bbq:
        rows = []
        keys = sorted({
            (r["category"], r["condition"], r["window"], r["arm"], r["alpha"])
            for r in bbq
        })
        for cat, cond, window, arm, alpha in keys:
            sel = [r for r in bbq if r["category"] == cat
                   and r["condition"] == cond and r["window"] == window
                   and r["arm"] == arm and r["alpha"] == alpha]
            # Polarity-balanced within block: the stereotype label sits on
            # opposite options in neg vs nonneg, so averaging the two cancels
            # any option/letter preference exactly (Week 2 finding 1).
            blocks: dict = {}
            for r in sel:
                bkey = (r["category"], r["example_id"] // 4)
                gap_clean = r["clean_lp"][0] - r["clean_lp"][1]
                gap_steer = r["steered_lp"][0] - r["steered_lp"][1]
                blocks.setdefault(bkey, {})[r["polarity"]] = {
                    "d_gap": gap_steer - gap_clean,
                    "pick_clean": int(np.argmax(r["clean_lp"])),
                    "pick_steer": int(np.argmax(r["steered_lp"])),
                    "kl": r["kl"],
                }
            paired = [v for v in blocks.values() if len(v) == 2]
            if not paired:
                continue
            d_gap = np.array([
                np.mean([v[p]["d_gap"] for p in ("neg", "nonneg")])
                for v in paired
            ])
            flips = np.array([
                np.mean([
                    float(v[p]["pick_steer"] != v[p]["pick_clean"])
                    for p in ("neg", "nonneg")
                ]) for v in paired
            ])
            lo, hi = boot_mean_ci(d_gap, seed=args.seed)
            rows.append({
                "category": cat, "condition": cond, "window": window,
                "arm": arm, "alpha": alpha, "n_blocks": len(paired),
                "mean_delta_gap": float(d_gap.mean()),
                "ci_delta_gap": [lo, hi],
                "answer_flip_rate": float(flips.mean()),
                "mean_kl": float(np.mean([
                    v[p]["kl"] for v in paired for p in v
                ])),
            })
        summary["bbq"] = rows
        print("\nBBQ steering (delta gap = change in logP(stereo letter) - "
              "logP(counter letter), polarity-balanced within block)")
        print(f"{'category':>18} {'cond':>9} {'window':>13} {'arm':>14} "
              f"{'alpha':>6} {'blk':>4} {'d_gap':>9} {'CI95':>18} {'flip':>7}")
        for r in rows:
            print(f"{r['category']:>18} {r['condition']:>9} {r['window']:>13} "
                  f"{r['arm']:>14} {r['alpha']:>6} {r['n_blocks']:>4} "
                  f"{r['mean_delta_gap']:>+9.3f} "
                  f"({r['ci_delta_gap'][0]:>+.2f},{r['ci_delta_gap'][1]:>+.2f})"
                  f" {r['answer_flip_rate']:>7.2%}")

    # ---- fluency cost ---------------------------------------------------
    flu = read_jsonl(out_dir / "fluency.jsonl")
    if flu:
        rows = []
        for arm in sorted({r["arm"] for r in flu}):
            for alpha in sorted({r["alpha"] for r in flu}):
                sel = [r for r in flu if r["arm"] == arm and r["alpha"] == alpha]
                if not sel:
                    continue
                d = np.array([r["steered_nll"] - r["clean_nll"] for r in sel])
                lo, hi = boot_mean_ci(d, seed=args.seed)
                rows.append({
                    "arm": arm, "alpha": alpha, "n": len(sel),
                    "mean_clean_nll": float(np.mean([r["clean_nll"] for r in sel])),
                    "mean_delta_nll": float(d.mean()),
                    "ci_delta_nll": [lo, hi],
                    "ppl_ratio": float(np.exp(d.mean())),
                })
        summary["fluency"] = rows
        print("\nFluency cost on held-out wikitext")
        print(f"{'arm':>14} {'alpha':>6} {'n':>4} {'clean_nll':>10} "
              f"{'d_nll':>8} {'ppl_ratio':>10}")
        for r in rows:
            print(f"{r['arm']:>14} {r['alpha']:>6} {r['n']:>4} "
                  f"{r['mean_clean_nll']:>10.3f} {r['mean_delta_nll']:>+8.3f} "
                  f"{r['ppl_ratio']:>10.3f}")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    make_plots(summary, out_dir)
    print(f"\nwrote {out_dir / 'summary.json'} and plots")


def make_plots(summary, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary.get("winogender"):
        rows = summary["winogender"]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
        for arm in sorted({r["arm"] for r in rows}):
            sel = sorted([r for r in rows if r["arm"] == arm],
                         key=lambda r: r["alpha"])
            a = [r["alpha"] for r in sel]
            axes[0].errorbar(
                a, [r["mean_delta_pref"] for r in sel],
                yerr=[
                    [r["mean_delta_pref"] - r["ci_delta_pref"][0] for r in sel],
                    [r["ci_delta_pref"][1] - r["mean_delta_pref"] for r in sel],
                ], fmt="-o", ms=3, capsize=2, label=arm,
            )
            axes[1].plot(a, [r["flip_rate"] for r in sel], "-o", ms=3, label=arm)
            axes[2].plot(a, [r["mean_kl"] for r in sel], "-o", ms=3, label=arm)
        for ax, ylab, title in [
            (axes[0], r"$\Delta$ [logP(she) - logP(he)]", "dose-response"),
            (axes[1], "pronoun preference flip rate", "sign flips"),
            (axes[2], "KL(steered || clean), nats", "distributional cost"),
        ]:
            ax.axhline(0, color="gray", ls=":", lw=1)
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(ylab)
            ax.set_title(title)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle("Winogender: steering the gender axis in the lens band")
        fig.tight_layout()
        fig.savefig(out_dir / "dose_response.png", dpi=130)
        plt.close(fig)

    if summary.get("bbq"):
        rows = summary["bbq"]
        cats = sorted({r["category"] for r in rows})
        fig, axes = plt.subplots(1, len(cats), figsize=(6.5 * len(cats), 4.4),
                                 squeeze=False)
        for ax, cat in zip(axes[0], cats, strict=True):
            for cond in ("ambig", "disambig"):
                for window in ("all", "pre_disambig"):
                    for arm in sorted({r["arm"] for r in rows}):
                        sel = sorted(
                            [r for r in rows if r["category"] == cat
                             and r["condition"] == cond
                             and r["window"] == window and r["arm"] == arm],
                            key=lambda r: r["alpha"],
                        )
                        if not sel:
                            continue
                        ax.plot([r["alpha"] for r in sel],
                                [r["mean_delta_gap"] for r in sel],
                                "-o" if window == "all" else "--s", ms=3,
                                label=f"{cond}/{window}/{arm}")
            ax.axhline(0, color="gray", ls=":", lw=1)
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$\Delta$ [logP(stereo) - logP(counter)]")
            ax.set_title(cat)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6)
        fig.suptitle("BBQ: gender-axis intervention, polarity-balanced")
        fig.tight_layout()
        fig.savefig(out_dir / "bbq_steering.png", dpi=130)
        plt.close(fig)

    if summary.get("fluency"):
        rows = summary["fluency"]
        fig, ax = plt.subplots(figsize=(6, 4.2))
        for arm in sorted({r["arm"] for r in rows}):
            sel = sorted([r for r in rows if r["arm"] == arm],
                         key=lambda r: r["alpha"])
            ax.plot([r["alpha"] for r in sel],
                    [r["mean_delta_nll"] for r in sel], "-o", ms=3, label=arm)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\Delta$ mean NLL (nats/token)")
        ax.set_title("collateral cost on held-out wikitext")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "fluency_cost.png", dpi=130)
        plt.close(fig)


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "mode",
        choices=["freq", "validate", "winogender", "bbq", "fluency",
                 "vignettes", "summarize"],
    )
    ap.add_argument("--out", default="analysis/out/week3/steering")
    ap.add_argument("--winogender", default="analysis/data/winogender_test.tsv")
    ap.add_argument("--bls", default="analysis/data/occupations-stats.tsv")
    ap.add_argument("--bbq-dir", default="/home/cataluna84/Workspace/BBQ")
    ap.add_argument("--categories", default="Gender_identity,Age")
    ap.add_argument("--blocks-per-cat", type=int, default=25)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--lens-repo", default="neuronpedia/jacobian-lens")
    ap.add_argument("--lens-revision", default="qwen-n1000")
    ap.add_argument(
        "--lens-file",
        default="qwen3.5-4b/jlens/Salesforce-wikitext/"
                "Qwen3.5-4B_jacobian_lens_n1000.pt",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--int8", action="store_true", default=True)
    ap.add_argument("--no-int8", dest="int8", action="store_false")
    ap.add_argument("--max-seq-len", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-freq-lines", type=int, default=1500)
    ap.add_argument("--pair", default=" he, she",
                    help="comma-separated token pair for the main axis")
    ap.add_argument("--alphas", default="0,0.5,1,1.5,2,3")
    ap.add_argument("--arms", default="counter,sham,random,counter_early")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()

    args.categories = [c for c in args.categories.split(",") if c]
    args.pair = [p for p in args.pair.split(",")]
    args.alphas = [float(a) for a in args.alphas.split(",") if a != ""]
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = set(args.arms) - set(ARM_SPECS)
    if unknown:
        raise SystemExit(f"unknown arms {sorted(unknown)}")

    {
        "freq": freq,
        "validate": validate,
        "winogender": steer_winogender,
        "bbq": steer_bbq,
        "fluency": fluency,
        "vignettes": vignettes,
        "summarize": summarize,
    }[args.mode](args)


if __name__ == "__main__":
    main()
