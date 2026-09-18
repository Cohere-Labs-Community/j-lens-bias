"""Build template-lens vectors for the subject model. Runs on the GPU box in the
`build` env, whose torch/transformers match the local environment.

For each word w (workspace paper, A.9.1):

    mu_w  = mean residual at the last position of passages ending just before w
    t_w   = (Sigma + lambda I)^-1 (mu_w - mu)

with mu and Sigma the residual mean and covariance over the same passages.

Each passage is run once as `passage + " " + phrase`. Causality makes the
residual at the last passage token identical to running the passage alone, and
the same pass gives the teacher-forced log-prob of the whole phrase -- a
plausibility check that works for multi-token and article-led phrases, where a
first-token check cannot ("a broken leg" starts with " a").

A held-out set per word is excluded from the fit and its residuals are saved.
It gives the out-of-sample readout (does a passage project most strongly onto its
own word's template?) and doubles as the CUDA-vs-MPS parity set. Sufficient
statistics are saved too, so lambda can be retuned locally without the GPU.

    python build_templates.py --vocab vocab.jsonl --passages out/passages --out out --pilot
"""

import argparse
import collections
import hashlib
import json
import re
import time
from pathlib import Path

import torch
import transformers

import jlens

from common import PINNED, SUBJECT_MODEL, leaks_word, slug

END_PUNCT = re.compile(r'[.!?"”]\s*$')


def check_versions(allow_mismatch: bool) -> dict:
    found = {"torch": torch.__version__.split("+")[0],
             "transformers": transformers.__version__}
    bad = {k: (found[k], PINNED[k]) for k in found if found[k] != PINNED[k]}
    if bad and not allow_mismatch:
        raise SystemExit(f"version mismatch (found, pinned): {bad} -- templates must be "
                         "fitted in the same torch/transformers as the local readout")
    return found


def string_verdict(text: str, phrase: str) -> str:
    if not text:
        return "empty"
    if leaks_word(text, phrase):
        return "leaks_word"
    if END_PUNCT.search(text):
        return "ends_with_punct"
    if not 12 <= len(text.split()) <= 120:
        return "length"
    return "ok"


def dedupe(recs: list[dict]) -> None:
    """Mark exact and near duplicates within one word (5-gram Jaccard >= 0.6)."""
    seen_exact, shingles = set(), []
    for r in recs:
        if r["verdict"] != "ok":
            continue
        norm = " ".join(r["text"].lower().split())
        if norm in seen_exact:
            r["verdict"] = "duplicate"
            continue
        w = norm.split()
        sh = {" ".join(w[i:i + 5]) for i in range(max(1, len(w) - 4))}
        if any(len(sh & s) / max(1, len(sh | s)) >= 0.6 for s in shingles):
            r["verdict"] = "near_duplicate"
            continue
        seen_exact.add(norm)
        shingles.append(sh)


def md5_int(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--passages", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--model", default=SUBJECT_MODEL)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lambda-rel", type=float, default=1.0,
                    help="ridge as a fraction of mean covariance eigenvalue; the notebook refits "
                         "from stats.pt, so this only sets the saved templates.pt")
    ap.add_argument("--holdout-per-word", type=int, default=5,
                    help="passages per word excluded from the fit and saved for testing")
    ap.add_argument("--min-kept", type=int, default=30)
    ap.add_argument("--hub-layer-frac", type=float, default=0.75)
    ap.add_argument("--limit-passages", type=int, default=None, help="smoke tests")
    ap.add_argument("--allow-version-mismatch", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    versions = check_versions(args.allow_version_mismatch)

    vocab = [json.loads(line) for line in open(args.vocab, encoding="utf-8")]
    if args.pilot:
        vocab = [e for e in vocab if e.get("pilot")]
    keys = [e["key"] for e in vocab]
    widx = {k: i for i, k in enumerate(keys)}
    meta = {e["key"]: e for e in vocab}

    # ---- load, string-filter, dedupe, choose held-out ------------------------
    recs = []
    for e in vocab:
        path = args.passages / f"{slug(e['key'])}.jsonl"
        if not path.exists():
            print(f"  missing passages for {e['key']!r}")
            continue
        word_recs = [json.loads(line) for line in open(path, encoding="utf-8")]
        for r in word_recs:
            r["text"] = r["text"].strip()
            r["verdict"] = string_verdict(r["text"], e["word"])
            if r["verdict"] == "ok" and r.get("judge_ok") is False:
                r["verdict"] = "judge_rejected"
            r["half"] = md5_int(r["text"]) % 2
            r["holdout"] = False
        dedupe(word_recs)
        ok = sorted((r for r in word_recs if r["verdict"] == "ok"),
                    key=lambda r: md5_int("holdout:" + r["text"]))
        for r in ok[:args.holdout_per_word]:
            r["holdout"] = True
        recs.extend(word_recs)
    if args.limit_passages:
        recs = recs[:args.limit_passages]
    todo = [r for r in recs if r["verdict"] == "ok"]
    print(f"{len(recs)} passages, {len(todo)} pass string filters "
          f"({sum(r['holdout'] for r in todo)} held out)")

    # ---- model -------------------------------------------------------------
    device = torch.device(args.device)
    acc_dev = device if device.type == "cuda" else torch.device("cpu")   # MPS has no float64
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(device)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf, tok)
    tok = model.tokenizer
    n_layers, d = model.n_layers, model.d_model
    layers = list(range(n_layers))
    W = len(keys)

    half_sums = torch.zeros(W * 2, n_layers, d, dtype=torch.float64, device=acc_dev)
    half_counts = torch.zeros(W * 2, dtype=torch.float64, device=acc_dev)
    g_sum = torch.zeros(n_layers, d, dtype=torch.float64, device=acc_dev)
    g_outer = torch.zeros(n_layers, d, d, dtype=torch.float64, device=acc_dev)
    g_n = 0

    # Tokenize once; drop passages whose tokens change when the phrase is
    # appended (a BPE merge across the boundary would move the read position).
    for r in todo:
        ids_p = tok(r["text"]).input_ids
        ids_f = tok(r["text"] + " " + meta[r["key"]]["word"]).input_ids
        if len(ids_f) <= len(ids_p) or ids_f[:len(ids_p)] != ids_p:
            r["verdict"] = "boundary_mismatch"
            continue
        r["_ids"], r["_pos"] = ids_f, len(ids_p) - 1
    todo = [r for r in todo if r["verdict"] == "ok"]
    todo.sort(key=lambda r: len(r["_ids"]))                 # less padding

    # Read positions are gathered inside the hooks, so every layer's full
    # [B, T, d] activation is never held at once.
    #
    # Move device and change dtype in TWO calls, never `.to(device, dtype)`.
    # On torch 2.14.0, `t.to("cpu", torch.float64)` from an MPS bfloat16 tensor
    # returns silently wrong values (off by ~150 on data with sd 3, sometimes
    # non-finite) -- MPS has no float64 -- while `.to("cpu").double()` is exact.
    copy_out = (lambda t: t.cpu()) if device.type == "mps" else (lambda t: t.clone())
    cur: dict = {}
    reads: dict = {}

    def make_hook(layer: int):
        def hook(module, inputs, output):
            h = output if torch.is_tensor(output) else output[0]
            reads[layer] = h[cur["rows"], cur["pos"]].to(acc_dev).double()
            if layer == n_layers - 1:                       # phrase positions, for plausibility
                reads["tail"] = copy_out(h[cur["rows_t"], cur["pos_t"]])
        return hook

    handles = [model.layers[l].register_forward_hook(make_hook(l)) for l in layers]
    held = {"keys": [], "texts": [], "phrases": [], "positions": [], "residuals": []}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    t0 = time.monotonic()
    with torch.inference_mode():
        for b0 in range(0, len(todo), args.batch):
            batch = todo[b0:b0 + args.batch]
            T = max(len(r["_ids"]) for r in batch)
            ids = torch.full((len(batch), T), pad, dtype=torch.long)
            mask = torch.zeros((len(batch), T), dtype=torch.long)
            for i, r in enumerate(batch):          # right padding: causal, so exact
                ids[i, :len(r["_ids"])] = torch.tensor(r["_ids"])
                mask[i, :len(r["_ids"])] = 1
            # positions predicting each phrase token: p .. p + n - 1
            spans = [(r["_pos"], len(r["_ids"]) - 1 - r["_pos"]) for r in batch]
            cur = {
                "rows": torch.arange(len(batch), device=device),
                "pos": torch.tensor([r["_pos"] for r in batch], device=device),
                "rows_t": torch.tensor([i for i, (_, n) in enumerate(spans) for _ in range(n)],
                                       device=device),
                "pos_t": torch.tensor([p + j for p, n in spans for j in range(n)], device=device),
            }
            reads.clear()
            model._text_module(input_ids=ids.to(device), attention_mask=mask.to(device),
                               use_cache=False)
            X = torch.stack([reads[l] for l in layers], dim=1)        # [B, L, d], float64

            # plausibility: teacher-forced log-prob of the whole phrase
            tail = reads["tail"].to(device)
            offset = 0
            for i, r in enumerate(batch):
                p, n = spans[i]
                tgt = torch.tensor(r["_ids"][p + 1:], device=device)
                logits = model.unembed(tail[offset:offset + n]).float()
                lp = torch.log_softmax(logits, dim=-1)[torch.arange(n, device=device), tgt]
                r["phrase_logprob"] = float(lp.sum())
                r["phrase_mean_logprob"] = float(lp.mean())
                r["first_token_rank"] = int((logits[0] > logits[0, tgt[0]]).sum())
                r["n_phrase_tokens"] = n
                offset += n

            # plausibility is recorded, never filtered on: a cutoff would keep only
            # contexts where the subject model already expects the word
            train = []
            for i, r in enumerate(batch):
                if r["holdout"]:
                    held["keys"].append(r["key"]); held["texts"].append(r["text"])
                    held["phrases"].append(meta[r["key"]]["word"])
                    held["positions"].append(r["_pos"])
                    held["residuals"].append(X[i].cpu().float())
                else:
                    train.append(i)
            for i, r in enumerate(batch):
                r["kept"] = i in train
            if not train:
                continue
            Xk = X[train]                                   # already float64 on acc_dev
            idx = torch.tensor([widx[batch[i]["key"]] * 2 + batch[i]["half"] for i in train],
                               device=acc_dev)
            half_sums.index_add_(0, idx, Xk)
            half_counts.index_add_(0, idx, torch.ones(len(train), dtype=torch.float64, device=acc_dev))
            g_sum += Xk.sum(0)
            g_outer += torch.einsum("bld,ble->lde", Xk, Xk)
            g_n += len(train)

            if (b0 // args.batch) % 20 == 0:
                rate = (b0 + len(batch)) / (time.monotonic() - t0)
                print(f"  {b0 + len(batch)}/{len(todo)} passages  {rate:.0f}/s", flush=True)
    for h in handles:
        h.remove()
    build_seconds = time.monotonic() - t0
    print(f"accumulated {g_n} passages; non-finite: sum={int((~torch.isfinite(g_sum)).sum())} "
          f"outer={int((~torch.isfinite(g_outer)).sum())}")

    # ---- templates -----------------------------------------------------------
    counts = half_counts.view(W, 2)
    sums = half_sums.view(W, 2, n_layers, d)
    n_w = counts.sum(1)
    mu = g_sum / g_n
    cov = g_outer / g_n - torch.einsum("ld,le->lde", mu, mu)
    eye = torch.eye(d, dtype=torch.float64, device=acc_dev)

    def solve(means: torch.Tensor) -> torch.Tensor:        # means [W, L, d] -> [L, W, d]
        out = torch.empty(n_layers, W, d, dtype=torch.float64, device=acc_dev)
        for l in range(n_layers):
            lam = args.lambda_rel * torch.trace(cov[l]) / d
            chol, info = torch.linalg.cholesky_ex(cov[l] + lam * eye)
            if int(info) != 0:
                diag = torch.diagonal(cov[l])
                raise SystemExit(
                    f"layer {l}: covariance + ridge not positive-definite (minor {int(info)}). "
                    f"n={g_n} trace={float(torch.trace(cov[l])):.3e} lambda={float(lam):.3e} "
                    f"min diag={float(diag.min()):.3e} max diag={float(diag.max()):.3e} "
                    f"non-finite={int((~torch.isfinite(cov[l])).sum())}")
            out[l] = torch.cholesky_solve((means[:, l] - mu[l]).T, chol).T
        return out

    safe = lambda c: c.clamp(min=1).view(W, 1, 1)
    templates = solve(sums.sum(1) / safe(n_w))
    t_a = solve(sums[:, 0] / safe(counts[:, 0]))
    t_b = solve(sums[:, 1] / safe(counts[:, 1]))
    split_cos = torch.nn.functional.cosine_similarity(t_a, t_b, dim=-1)    # [L, W]
    both_halves = (counts > 0).all(1)

    # ---- out-of-sample readout on the held-out set ---------------------------
    holdout = {"layers": [], "top1": [], "top5": [], "median_own_rank": []}
    hubs = []
    live = (n_w > 0).cpu()
    if held["keys"]:
        H = torch.stack(held["residuals"])                          # [N, L, d] float32
        own = torch.tensor([widx[k] for k in held["keys"]])
        hub_layer = round(args.hub_layer_frac * (n_layers - 1))
        for l in layers:
            scores = (H[:, l].double() - mu[l].cpu()) @ templates[l].cpu().T   # [N, W]
            scores[:, ~live] = float("-inf")
            own_score = scores[torch.arange(len(own)), own]
            rank = (scores > own_score[:, None]).sum(1)
            holdout["layers"].append(l)
            holdout["top1"].append(round(float((rank == 0).float().mean()), 4))
            holdout["top5"].append(round(float((rank < 5).float().mean()), 4))
            holdout["median_own_rank"].append(int(rank.median()))
            if l == hub_layer:
                k = min(10, int(live.sum()))
                top = scores.topk(k, dim=1).indices.flatten().tolist()
                expected = k / max(1, int(live.sum()))
                freq = collections.Counter(top)
                hubs = [{"key": keys[j], "share_of_passages": round(c / len(own), 3),
                         "vs_expected": round(c / len(own) / expected, 1)}
                        for j, c in freq.most_common(15)]
        held["residuals"] = H.half() if not args.pilot else H
        torch.save(held, args.out / "holdout.pt")

    torch.save({"keys": keys, "layers": layers, "lambda_rel": args.lambda_rel,
                "templates": templates.cpu().float()}, args.out / "templates.pt")
    torch.save({"keys": keys, "layers": layers, "n": g_n,
                "half_sums": sums.cpu().float(), "half_counts": counts.cpu(),
                "mean": mu.cpu().float(), "cov": cov.cpu().float()}, args.out / "stats.pt")
    with open(args.out / "passage_verdicts.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")},
                                ensure_ascii=False) + "\n")

    # ---- summary -------------------------------------------------------------
    by_source = collections.defaultdict(collections.Counter)
    lp_by_source = collections.defaultdict(list)
    for r in recs:
        tag = f"{meta[r['key']]['source']}/{meta[r['key']].get('hb_preference') or '-'}"
        by_source[tag][r["verdict"]] += 1
        if "phrase_mean_logprob" in r:
            lp_by_source[tag].append(r["phrase_mean_logprob"])
    q = lambda v: {p: round(float(torch.tensor(v).quantile(p)), 2) for p in (0.1, 0.5, 0.9)}
    kept_per_word = {k: int(n_w[i]) for i, k in enumerate(keys)}
    summary = {
        "versions": versions, "model": args.model, "n_layers": n_layers, "d_model": d,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else args.device,
        "passages": len(recs), "kept_for_fit": g_n, "held_out": len(held["keys"]),
        "verdicts": dict(collections.Counter(r["verdict"] for r in recs)),
        "verdicts_by_source": {k: dict(v) for k, v in by_source.items()},
        "phrase_mean_logprob_by_source": {k: q(v) for k, v in lp_by_source.items()},
        "kept_per_word": kept_per_word,
        "words_below_min_kept": [k for k, v in kept_per_word.items() if v < args.min_kept],
        "split_half_cosine_median_by_layer": [
            round(float(split_cos[l][both_halves].median()), 4) if both_halves.any() else None
            for l in layers],
        "holdout_readout": holdout,
        "hubs_at_layer": hubs,
        "build_seconds": round(build_seconds, 1),
        "passages_per_second": round(len(todo) / max(build_seconds, 1e-9), 1),
    }
    (args.out / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("kept_per_word", "verdicts_by_source")}, indent=2))


if __name__ == "__main__":
    main()
