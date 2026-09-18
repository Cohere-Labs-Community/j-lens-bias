#!/usr/bin/env python3
"""Week 3, Option 1: continuous-ground-truth gender readout on Winogender.

Why this dataset. Week 2 (mine and every other submission in the study group)
read the lens at an answer slot and compared the ranks of two *group phrases*
in BBQ. Those phrases are multi-token and unequally frequent, so the resulting
"rank gap" mixes bias with tokenization: one participant's effect vanished on
the single-token subset, two others collapsed onto the rank of ``' The'``, and
the most developed submission declared its own rank arm non-interpretable
after the pooled effect moved between +0.26 and -0.01 across resamples.

Winogender removes that confound by construction:

* The tracked tokens are the *same three pronouns for every item* (all single
  tokens in this tokenizer), so token frequency, length and tokenization are
  held exactly fixed across items. Only the occupation varies.
* The task is a cloze continuation, not multiple choice: the prompt is cut
  immediately before the pronoun and the readout is the next-token
  distribution. No option letters, no ``Answer:`` scaffold.
* Ground truth is *continuous*: the Winogender release ships US Bureau of
  Labor Statistics percent-female figures per occupation, so the hypothesis
  becomes a correlation over 60 occupations rather than a binary
  stereotype/counter contrast.
* The dataset supplies its own negative control: in half the templates the
  pronoun refers to the *participant* (customer, client, ...), not the
  occupation, so occupational stereotype should predict the readout much less
  there while everything else about the sentence stays the same.

Design
------
240 templates (60 occupations x 2 participant variants x 2 antecedents), each
with a male/female/neutral variant that shares an identical prefix. Per
template we run one forward pass and read out every layer at the final
position, recording the log-probability and full-vocabulary rank of the
case-matched pronoun triple (he/she/they, his/her/their, him/her/them).

Primary statistic ``pref = logP(female form) - logP(male form)``; secondary
``hedge = logP(neutral form) - max(logP(male), logP(female))``.

Four arms, chosen to be the controls nobody in the group ran:

==================  ====================================================
arm                 what it tests
==================  ====================================================
raw/jacobian        the measurement
raw/logit           is this the *Jacobian* transport, or just the
                    residual stream? (``use_jacobian=False``)
raw/permuted        norm- and spectrum-matched random transport: J with
                    its rows and columns permuted. Any linear map of the
                    same size would produce this much.
chat/jacobian       does a wikitext-fitted lens survive chat formatting?
==================  ====================================================

Modes: ``collect`` (resumable record writer), ``fidelity`` (int8-GPU vs
CPU-bf16 agreement on the actual statistic), ``summarize`` (tables, plots,
permutation tests, BH-FDR).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers

import jlens
from jlens.vis import _ranks_of

# Winogender uses one pronoun per sentence in one of three cases. The male
# surface form identifies the case; the triple is (male, female, neutral).
CASE_TRIPLES = {
    "he": ("he", "she", "they"),
    "his": ("his", "her", "their"),
    "him": ("him", "her", "them"),
}

# Measured in Week 2 on the same model+lens: the stereotype signal in BBQ was
# flat at 0.500 through L16, rose from L20 and peaked at L23-L25.
BAND = list(range(20, 31))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_templates(tsv_path: Path) -> list[dict]:
    """Group the 720 Winogender sentences into 240 cloze templates.

    Each template is one (occupation, participant, antecedent) triple whose
    male/female/neutral variants differ only in the pronoun. The prefix is the
    sentence truncated immediately before the pronoun; it is asserted
    identical across the three variants (holds for all 240).
    """
    rows = list(csv.DictReader(tsv_path.open(encoding="utf-8"), delimiter="\t"))
    groups: dict[tuple, dict] = {}
    for r in rows:
        key = (r["occupation"], r["participant"], r["label"])
        groups.setdefault(key, {})[r["gender"]] = r

    templates = []
    for (occ, part, label), by_gender in sorted(groups.items()):
        if set(by_gender) != {"male", "female", "neutral"}:
            raise ValueError(f"incomplete template {occ}/{part}/{label}")
        male = by_gender["male"]
        case = male["pronoun"].lower()
        if case not in CASE_TRIPLES:
            raise ValueError(f"unexpected male pronoun {case!r}")
        prefixes = set()
        for row in by_gender.values():
            form = row["pronoun"]
            idx = _pronoun_index(row["sentence"], form)
            prefixes.add(row["sentence"][:idx].rstrip())
        if len(prefixes) != 1:
            raise ValueError(f"prefix mismatch for {occ}/{part}/{label}")
        templates.append(
            {
                "tid": f"{occ}.{part}.{label}",
                "occupation": occ,
                "participant": part,
                "label": int(label),
                "target": male["target"],
                "target_kind": (
                    "occupation" if male["target"] == occ else "participant"
                ),
                "case": case,
                "forms": list(CASE_TRIPLES[case]),
                "prefix": prefixes.pop(),
                "sentence_male": male["sentence"],
            }
        )
    return templates


def _pronoun_index(sentence: str, form: str) -> int:
    """Character offset of the pronoun word inside the sentence."""
    import re

    match = re.search(rf"\b{re.escape(form)}\b", sentence, re.IGNORECASE)
    if match is None:
        raise ValueError(f"pronoun {form!r} not found in {sentence!r}")
    return match.start()


def load_bls(tsv_path: Path) -> dict[str, dict]:
    """Occupation -> {bls_pct_female, bergsma_pct_female} from the Winogender
    release (US BLS 2015 figures plus Bergsma & Lin's web-corpus estimates)."""
    out = {}
    for r in csv.DictReader(tsv_path.open(encoding="utf-8"), delimiter="\t"):
        out[r["occupation"]] = {
            "bls": float(r["bls_pct_female"]),
            "bergsma": float(r["bergsma_pct_female"]),
        }
    return out


def build_prompt(template: dict, style: str, tokenizer) -> str:
    """``raw`` is the bare cloze prefix; ``chat`` wraps it in the model's chat
    template with the prefix prefilled as the start of the reply, so the
    readout position is the same kind of slot in both arms."""
    prefix = template["prefix"]
    if style == "raw":
        return prefix
    if style == "chat":
        messages = [
            {
                "role": "user",
                "content": f"Complete this sentence:\n{prefix} ...",
            }
        ]
        try:
            head = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            head = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return head + prefix
    raise ValueError(f"unknown prompt style {style!r}")


# --------------------------------------------------------------------------
# Model / lens
# --------------------------------------------------------------------------

class PermutedLens(jlens.JacobianLens):
    """Norm- and spectrum-matched random transport control.

    ``J_l`` has its rows and columns permuted by a fixed random permutation.
    Frobenius norm and singular values are preserved exactly; the alignment
    between residual directions and vocabulary directions is destroyed. If the
    Jacobian lens's readout is doing real work, this arm should not reproduce
    it.
    """

    def __init__(self, lens: jlens.JacobianLens, seed: int = 0) -> None:
        super().__init__(
            lens.jacobians, n_prompts=lens.n_prompts, d_model=lens.d_model
        )
        gen = torch.Generator().manual_seed(seed)
        self.perm_rows = torch.randperm(lens.d_model, generator=gen)
        self.perm_cols = torch.randperm(lens.d_model, generator=gen)

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        J = self.jacobians[layer].to(residual.device)
        J = J[self.perm_rows.to(J.device)][:, self.perm_cols.to(J.device)]
        return residual @ J.T


def load_model_and_lens(args):
    print(f"loading {args.model} ...", flush=True)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    if args.device == "cuda" and args.int8:
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["lm_head", "embed_tokens"],
        )
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config, device_map={"": 0}
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


def pronoun_ids(tokenizer) -> dict[str, int]:
    """Leading-space pronoun forms, asserted single-token."""
    ids = {}
    for forms in CASE_TRIPLES.values():
        for form in forms:
            enc = tokenizer(" " + form, add_special_tokens=False).input_ids
            if len(enc) != 1:
                raise ValueError(
                    f"' {form}' is not a single token: {enc} "
                    f"{[tokenizer.decode([i]) for i in enc]}"
                )
            ids[form] = enc[0]
    return ids


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------

ARMS = [("raw", "jacobian"), ("raw", "logit"), ("raw", "permuted"),
        ("chat", "jacobian")]


def collect(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"

    templates = load_templates(Path(args.winogender))
    if args.limit:
        templates = templates[: args.limit]
    arms = [a for a in ARMS if f"{a[0]}/{a[1]}" in args.arms]
    print(f"{len(templates)} templates x {len(arms)} arms "
          f"= {len(templates) * len(arms)} lens calls", flush=True)

    done = set()
    if records_path.exists() and args.resume:
        with records_path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                done.add((rec["tid"], rec["style"], rec["transport"]))
        print(f"resuming: {len(done)} records already present", flush=True)

    model, tokenizer, lens = load_model_and_lens(args)
    pids = pronoun_ids(tokenizer)
    print("pronoun token ids:", pids, flush=True)
    lenses = {"jacobian": lens, "logit": lens}
    if any(t == "permuted" for _, t in arms):
        lenses["permuted"] = PermutedLens(lens, seed=args.seed)
    final_layer = model.n_layers - 1
    # For the sanity curve: is the readout at this slot in "pronoun mode" at
    # all? Rank of the best of all eight pronoun forms.
    all_pron = torch.tensor(sorted(set(pids.values())), dtype=torch.long)

    fh = records_path.open("a", encoding="utf-8")
    t0, n_run = time.time(), 0
    for idx, template in enumerate(templates):
        for style, transport in arms:
            if (template["tid"], style, transport) in done:
                continue
            prompt = build_prompt(template, style, tokenizer)
            lens_logits, model_logits, input_ids = lenses[transport].apply(
                model,
                prompt,
                positions=[-1],
                max_seq_len=args.max_seq_len,
                use_jacobian=(transport != "logit"),
            )
            vec = torch.tensor(
                [pids[f] for f in template["forms"]], dtype=torch.long
            )
            layer_logits = dict(lens_logits)
            layer_logits[final_layer] = model_logits

            layer_recs = []
            for layer, logits_cpu in sorted(layer_logits.items()):
                logits = logits_cpu.to(args.device)
                ranks = _ranks_of(logits, vec.to(logits.device))[0].tolist()
                logprobs = torch.log_softmax(logits[0].float(), dim=-1)
                lp = [float(logprobs[int(i)]) for i in vec]
                pron_ranks = _ranks_of(logits, all_pron.to(logits.device))[0]
                best = int(pron_ranks.min())
                layer_recs.append(
                    {
                        "layer": int(layer),
                        "model": bool(layer == final_layer),
                        "lp": lp,
                        "rk": ranks,
                        "best_pron_rank": best,
                        "best_pron": tokenizer.decode(
                            [int(all_pron[int(pron_ranks.argmin())])]
                        ),
                    }
                )
                del logits, logprobs

            rec = {
                "tid": template["tid"],
                "style": style,
                "transport": transport,
                "occupation": template["occupation"],
                "participant": template["participant"],
                "target_kind": template["target_kind"],
                "case": template["case"],
                "forms": template["forms"],
                "prefix": template["prefix"],
                "seq_len": int(input_ids.shape[1]),
                "token_ids": [int(pids[f]) for f in template["forms"]],
                "layers": layer_recs,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            del lens_logits, model_logits
            n_run += 1

        if (idx + 1) % 10 == 0 and n_run:
            rate = (time.time() - t0) / n_run
            print(f"  template {idx + 1}/{len(templates)}  "
                  f"{rate:.2f}s/call  ({n_run} new)", flush=True)
    fh.close()
    print(f"done -> {records_path}", flush=True)


# --------------------------------------------------------------------------
# fidelity
# --------------------------------------------------------------------------

def fidelity(args) -> None:
    """int8-GPU vs CPU-bf16 agreement on the actual statistic.

    Week 2 checked rank agreement; here the statistic is a log-probability
    difference, so we compare ``pref`` values directly (Spearman plus mean
    absolute difference in nats) as well as rank agreement.
    """
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    templates = load_templates(Path(args.winogender))
    templates = [t for t in templates if t["target_kind"] == "occupation"]
    templates = templates[: args.limit or 8]
    layer_subset = list(range(0, 31, 5))

    results = {}
    for tag, device, int8 in [("cpu_bf16", "cpu", False),
                              ("gpu_int8", "cuda", True)]:
        sub = argparse.Namespace(**vars(args))
        sub.device, sub.int8 = device, int8
        model, tokenizer, lens = load_model_and_lens(sub)
        pids = pronoun_ids(tokenizer)
        runs = []
        for template in templates:
            prompt = build_prompt(template, "raw", tokenizer)
            lens_logits, model_logits, _ = lens.apply(
                model, prompt, positions=[-1], layers=layer_subset
            )
            vec = torch.tensor([pids[f] for f in template["forms"]])
            per_layer = {}
            layer_logits = dict(lens_logits)
            layer_logits[model.n_layers - 1] = model_logits
            for layer, logits in layer_logits.items():
                lg = torch.log_softmax(logits[0].float(), dim=-1)
                lp = [float(lg[int(i)]) for i in vec]
                rk = _ranks_of(logits, vec)[0].tolist()
                per_layer[layer] = {"pref": lp[1] - lp[0], "rk": rk}
            runs.append(per_layer)
            del lens_logits, model_logits
        results[tag] = runs
        del model, lens
        import gc

        gc.collect()
        torch.cuda.empty_cache()

    from scipy.stats import spearmanr

    report = {}
    for layer in sorted(results["cpu_bf16"][0]):
        a = [r[layer]["pref"] for r in results["cpu_bf16"]]
        b = [r[layer]["pref"] for r in results["gpu_int8"]]
        sign = sum(int(np.sign(x) == np.sign(y)) for x, y in zip(a, b, strict=True))
        rho = float(spearmanr(a, b).statistic) if len(set(a)) > 1 else float("nan")
        report[layer] = {
            "spearman_pref": rho,
            "mean_abs_diff_nats": float(np.mean(np.abs(np.array(a) - np.array(b)))),
            "sign_agree": sign / len(a),
        }
        print(f"layer {layer:>3}: spearman(pref)={rho:.4f}  "
              f"mean|dPref|={report[layer]['mean_abs_diff_nats']:.4f} nats  "
              f"sign-agree={sign}/{len(a)}", flush=True)
    (out_dir / "fidelity.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {out_dir / 'fidelity.json'}")


# --------------------------------------------------------------------------
# summarize
# --------------------------------------------------------------------------

def iter_records(out_dir: Path):
    with (out_dir / "records.jsonl").open() as fh:
        for line in fh:
            yield json.loads(line)


def spearman(x, y) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def perm_p(x, y, n: int = 10000, seed: int = 0) -> float:
    """Two-sided permutation p-value for Spearman rho, shuffling the labels."""
    rng = np.random.default_rng(seed)
    obs = abs(spearman(x, y))
    y = np.asarray(y, dtype=float)
    hits = 0
    for _ in range(n):
        if abs(spearman(x, rng.permutation(y))) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n + 1)


def boot_ci_rho(x, y, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap CI for Spearman rho, resampling occupations (the unit)."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        if len(set(x[idx])) < 3:
            continue
        vals.append(spearman(x[idx], y[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bh_fdr(pvals: list[float], q: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg: which hypotheses survive at FDR q."""
    n = len(pvals)
    order = np.argsort(pvals)
    keep = np.zeros(n, dtype=bool)
    thresh = 0
    for rank, i in enumerate(order, start=1):
        if pvals[i] <= q * rank / n:
            thresh = rank
    for rank, i in enumerate(order, start=1):
        if rank <= thresh:
            keep[i] = True
    return keep.tolist()


def layer_stat(rec: dict, layer: int) -> dict:
    return next(x for x in rec["layers"] if x["layer"] == layer)


def summarize(args) -> None:
    out_dir = Path(args.out)
    recs = list(iter_records(out_dir))
    print(f"{len(recs)} records")
    bls = load_bls(Path(args.bls))
    layers = sorted({lr["layer"] for r in recs for lr in r["layers"]})
    model_layer = max(layers)

    def pref(rec, layer):
        lp = layer_stat(rec, layer)["lp"]
        return lp[1] - lp[0]

    def hedge(rec, layer):
        lp = layer_stat(rec, layer)["lp"]
        return lp[2] - max(lp[0], lp[1])

    # ---- sanity: is this slot in "pronoun mode" at all? -------------------
    sanity = []
    for layer in layers:
        sel = [r for r in recs if r["style"] == "raw"
               and r["transport"] == "jacobian"]
        ranks = np.array([layer_stat(r, layer)["best_pron_rank"] for r in sel])
        tgt = np.array([
            min(layer_stat(r, layer)["rk"][:2]) for r in sel
        ])
        sanity.append({
            "layer": layer,
            "is_model_row": layer == model_layer,
            "n": len(sel),
            "median_best_pronoun_rank": float(np.median(ranks)),
            "frac_pronoun_top1": float(np.mean(ranks == 0)),
            "frac_pronoun_top10": float(np.mean(ranks < 10)),
            "median_best_gendered_rank": float(np.median(tgt)),
        })
    (out_dir / "sanity.json").write_text(json.dumps(sanity, indent=1))
    print("\nsanity curve (raw/jacobian): median rank of the best pronoun")
    for row in sanity:
        if row["layer"] % 4 == 0 or row["is_model_row"]:
            print(f"  L{row['layer']:>2}{' (model)' if row['is_model_row'] else '':>8}"
                  f"  median_best_pronoun_rank={row['median_best_pronoun_rank']:>9.1f}"
                  f"  top1={row['frac_pronoun_top1']:.2%}")

    # ---- BLS correlation trajectory, per arm and antecedent ---------------
    corr_rows = []
    for style, transport in ARMS:
        for kind in ("occupation", "participant"):
            sel = [r for r in recs if r["style"] == style
                   and r["transport"] == transport
                   and r["target_kind"] == kind]
            if not sel:
                continue
            for layer in layers:
                by_occ: dict[str, list[float]] = {}
                for r in sel:
                    by_occ.setdefault(r["occupation"], []).append(pref(r, layer))
                occs = sorted(by_occ)
                lean = [float(np.mean(by_occ[o])) for o in occs]
                row = {
                    "style": style, "transport": transport, "target_kind": kind,
                    "layer": layer, "is_model_row": layer == model_layer,
                    "n_occ": len(occs),
                    "mean_pref": float(np.mean(lean)),
                }
                for gt in ("bls", "bergsma"):
                    truth = [bls[o][gt] for o in occs]
                    rho = spearman(lean, truth)
                    lo, hi = boot_ci_rho(lean, truth, seed=args.seed)
                    row[f"rho_{gt}"] = rho
                    row[f"ci_{gt}"] = [lo, hi]
                    row[f"p_{gt}"] = perm_p(
                        lean, truth, n=args.n_perm, seed=args.seed
                    )
                corr_rows.append(row)

    # BH-FDR over the layer-wise permutation tests, within each arm/antecedent
    for style, transport in ARMS:
        for kind in ("occupation", "participant"):
            grp = [r for r in corr_rows if r["style"] == style
                   and r["transport"] == transport and r["target_kind"] == kind]
            if not grp:
                continue
            for gt in ("bls", "bergsma"):
                keep = bh_fdr([r[f"p_{gt}"] for r in grp], q=0.05)
                for r, k in zip(grp, keep, strict=True):
                    r[f"fdr_{gt}"] = bool(k)
    (out_dir / "bls_correlation.json").write_text(json.dumps(corr_rows, indent=1))

    print("\nSpearman(lens pronoun preference, BLS % female) by layer")
    print(f"{'arm':>18} {'antecedent':>12} " +
          " ".join(f"{'L' + str(p):>16}" for p in BAND[::3] + [model_layer]))
    for style, transport in ARMS:
        for kind in ("occupation", "participant"):
            cells = []
            for layer in BAND[::3] + [model_layer]:
                row = next((r for r in corr_rows if r["style"] == style
                            and r["transport"] == transport
                            and r["target_kind"] == kind
                            and r["layer"] == layer), None)
                if row is None:
                    cells.append(f"{'-':>16}")
                    continue
                star = "*" if row.get("fdr_bls") else " "
                cells.append(f"{row['rho_bls']:>+.3f} p={row['p_bls']:.4f}{star}")
            print(f"{style + '/' + transport:>18} {kind:>12} " + " ".join(cells))

    # ---- band statistic and per-occupation table -------------------------
    band = [l for l in BAND if l in layers]
    occ_rows = []
    sel = [r for r in recs if r["style"] == "raw" and r["transport"] == "jacobian"]
    for kind in ("occupation", "participant"):
        by_occ: dict[str, list[dict]] = {}
        for r in sel:
            if r["target_kind"] == kind:
                by_occ.setdefault(r["occupation"], []).append(r)
        for occ, rs in sorted(by_occ.items()):
            occ_rows.append({
                "occupation": occ, "target_kind": kind, "n": len(rs),
                "bls": bls[occ]["bls"], "bergsma": bls[occ]["bergsma"],
                "band_pref": float(np.mean(
                    [np.mean([pref(r, l) for l in band]) for r in rs])),
                "model_pref": float(np.mean([pref(r, model_layer) for r in rs])),
                "band_hedge": float(np.mean(
                    [np.mean([hedge(r, l) for l in band]) for r in rs])),
                "model_hedge": float(np.mean(
                    [hedge(r, model_layer) for r in rs])),
            })
    (out_dir / "occupation_lean.json").write_text(json.dumps(occ_rows, indent=1))

    # ---- lens-behaviour gap ---------------------------------------------
    lb = {}
    for kind in ("occupation", "participant"):
        rows = [r for r in occ_rows if r["target_kind"] == kind]
        band_v = [r["band_pref"] for r in rows]
        model_v = [r["model_pref"] for r in rows]
        truth = [r["bls"] for r in rows]
        lb[kind] = {
            "n_occ": len(rows),
            "rho_band_bls": spearman(band_v, truth),
            "rho_model_bls": spearman(model_v, truth),
            "rho_band_model": spearman(band_v, model_v),
            "mean_band_pref": float(np.mean(band_v)),
            "mean_model_pref": float(np.mean(model_v)),
            "ci_band_bls": list(boot_ci_rho(band_v, truth, seed=args.seed)),
            "ci_model_bls": list(boot_ci_rho(model_v, truth, seed=args.seed)),
        }
        print(f"\nlens-behaviour gap [{kind}]: rho(band,BLS)="
              f"{lb[kind]['rho_band_bls']:+.3f}  rho(model,BLS)="
              f"{lb[kind]['rho_model_bls']:+.3f}  rho(band,model)="
              f"{lb[kind]['rho_band_model']:+.3f}")
    (out_dir / "lens_behaviour.json").write_text(json.dumps(lb, indent=1))

    # ---- hedging trajectory ---------------------------------------------
    hedge_rows = []
    for layer in layers:
        for kind in ("occupation", "participant"):
            ss = [r for r in sel if r["target_kind"] == kind]
            vals = np.array([hedge(r, layer) for r in ss])
            hedge_rows.append({
                "layer": layer, "target_kind": kind, "n": len(ss),
                "mean_hedge": float(vals.mean()),
                "frac_neutral_wins": float(np.mean(vals > 0)),
            })
    (out_dir / "hedging.json").write_text(json.dumps(hedge_rows, indent=1))

    make_plots(corr_rows, sanity, occ_rows, hedge_rows, layers, out_dir)
    print(f"\nwrote tables and plots under {out_dir}")


def make_plots(corr_rows, sanity, occ_rows, hedge_rows, layers, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_layer = max(layers)

    # correlation trajectory, one line per arm
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, kind in zip(axes, ("occupation", "participant"), strict=True):
        for style, transport in ARMS:
            rows = [r for r in corr_rows if r["style"] == style
                    and r["transport"] == transport and r["target_kind"] == kind]
            if not rows:
                continue
            rows.sort(key=lambda r: r["layer"])
            ax.plot([r["layer"] for r in rows], [r["rho_bls"] for r in rows],
                    "-o", ms=3, label=f"{style}/{transport}")
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.axvspan(BAND[0] - 0.5, BAND[-1] + 0.5, color="orange", alpha=0.08)
        ax.set_xlabel("layer (last point = model output row)")
        ax.set_title(f"antecedent = {kind}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"Spearman $\rho$(pronoun preference, BLS % female)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Does the lens readout track real-world occupational gender "
                 "statistics?")
    fig.tight_layout()
    fig.savefig(out_dir / "bls_correlation.png", dpi=130)
    plt.close(fig)

    # scatter at the band and at the model row
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    rows = [r for r in occ_rows if r["target_kind"] == "occupation"]
    for ax, key, title in [
        (axes[0], "band_pref", f"lens band L{BAND[0]}-L{BAND[-1]}"),
        (axes[1], "model_pref", "model output row"),
    ]:
        ax.scatter([r["bls"] for r in rows], [r[key] for r in rows], s=18)
        for r in rows:
            if r["bls"] > 85 or r["bls"] < 8 or abs(r[key]) > 3:
                ax.annotate(r["occupation"], (r["bls"], r[key]), fontsize=6,
                            alpha=0.7)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_xlabel("BLS % female in occupation")
        ax.set_ylabel("logP(she) - logP(he)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "bls_scatter.png", dpi=130)
    plt.close(fig)

    # sanity + hedging
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].semilogy([r["layer"] for r in sanity],
                     [max(r["median_best_pronoun_rank"], 0.5) for r in sanity],
                     "-o", ms=3)
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("median rank of best pronoun (log)")
    axes[0].set_title("precondition: is the slot in pronoun mode?")
    axes[0].grid(alpha=0.3)
    for kind, style in (("occupation", "-o"), ("participant", "--s")):
        rows = [r for r in hedge_rows if r["target_kind"] == kind]
        rows.sort(key=lambda r: r["layer"])
        axes[1].plot([r["layer"] for r in rows], [r["mean_hedge"] for r in rows],
                     style, ms=3, label=kind)
    axes[1].axhline(0, color="gray", ls=":", lw=1)
    axes[1].set_xlabel("layer")
    axes[1].set_ylabel("logP(they) - max(logP(he), logP(she))")
    axes[1].set_title("hedging: does the neutral pronoun win?")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "sanity_hedging.png", dpi=130)
    plt.close(fig)
    print(f"model row = L{model_layer}")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["collect", "fidelity", "summarize"])
    ap.add_argument("--winogender", default="analysis/data/winogender_test.tsv")
    ap.add_argument("--bls", default="analysis/data/occupations-stats.tsv")
    ap.add_argument("--out", default="analysis/out/week3/winogender")
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
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument(
        "--arms",
        default="raw/jacobian,raw/logit,raw/permuted,chat/jacobian",
        help="comma-separated style/transport arms to collect",
    )
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()
    args.arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    {"collect": collect, "fidelity": fidelity, "summarize": summarize}[args.mode](
        args
    )


if __name__ == "__main__":
    main()
