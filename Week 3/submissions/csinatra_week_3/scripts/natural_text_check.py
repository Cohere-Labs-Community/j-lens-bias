"""Do templates built from generated passages recognize the same words in natural text?

Finds real occurrences of each vocabulary word in WikiText-103, keeps the text
just before the word, reads the residual at its last token (the same read
position the templates are fitted on) and scores it against every template.
Held-out generated passages for the same words are the in-distribution baseline.
Runs locally.

    python natural_text_check.py --run ../../_template_lens/out_pilot2 \
        --vocab ../../_template_lens/vocab_pilot2.jsonl
"""

import argparse
import glob
import hashlib
import heapq
import json
import re
from pathlib import Path

import pandas as pd
import torch
import transformers

import jlens

from common import SUBJECT_MODEL

WIKITEXT = ("~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/"
            "wikitext-103-raw-v1/train-*.parquet")


def detok(s: str) -> str:
    """Undo WikiText's spaced tokenization ("78 @-@ year", "word ,")."""
    s = re.sub(r" @(.)@ ", r"\1", s)
    s = re.sub(r" ([,.;:!?)\]'])", r"\1", s)
    s = re.sub(r"([(\[]) ", r"\1", s)
    return s.replace(" n't", "n't").replace(" 's", "'s")


def find_contexts(vocab: list[dict], live: set[str], per_word: int, context_words: int,
                  min_words: int) -> pd.DataFrame:
    """Up to per_word occurrences per word, chosen by hash so the sample is fixed."""
    surface_to_key = {v: e["key"] for e in vocab if e["key"] in live for v in e["variants"]}
    alts = sorted(surface_to_key, key=len, reverse=True)
    pattern = re.compile(r"(?<=\s)(" + "|".join(map(re.escape, alts)) + r")(?![\w-])")
    heaps: dict[str, list] = {k: [] for k in live}
    files = sorted(glob.glob(str(Path(WIKITEXT).expanduser())))
    for f in files:
        for pi, raw in enumerate(pd.read_parquet(f).text):
            if not raw.strip() or raw.lstrip().startswith("="):
                continue
            para = detok(raw.strip())
            for m in pattern.finditer(para):
                context = para[:m.start()].rstrip()
                words = context.split()
                if len(words) < min_words:
                    continue
                context = " ".join(words[-context_words:])
                key = surface_to_key[m.group(1)]
                h = -int(hashlib.md5(f"{key}:{f}:{pi}:{m.start()}".encode()).hexdigest()[:15], 16)
                item = (h, context, m.group(1))
                if len(heaps[key]) < per_word:
                    heapq.heappush(heaps[key], item)
                elif h > heaps[key][0][0]:
                    heapq.heapreplace(heaps[key], item)
    rows = [{"key": k, "surface": s, "text": c,
             "word_in_context": s.lower() in c.lower()}
            for k, items in heaps.items() for (_, c, s) in items]
    return pd.DataFrame(rows).sort_values(["key", "text"]).reset_index(drop=True)


def read_residuals(texts: list[str], model_name: str, device: str) -> torch.Tensor:
    dev = torch.device(device)
    hf = transformers.AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(dev)
    model = jlens.from_hf(hf, transformers.AutoTokenizer.from_pretrained(model_name))
    tok, layers = model.tokenizer, list(range(model.n_layers))
    got: dict = {}
    hooks = [model.layers[l].register_forward_hook(
        lambda m, i, o, l=l: got.__setitem__(l, (o if torch.is_tensor(o) else o[0])))
        for l in layers]
    out = torch.empty(len(texts), len(layers), hf.config.get_text_config().hidden_size)
    with torch.inference_mode():
        for k, text in enumerate(texts):
            ids = tok(text, return_tensors="pt").input_ids
            got.clear()
            model._text_module(input_ids=ids.to(dev), use_cache=False)    # one at a time: no padding
            # two-step move: never .to(device, dtype) from MPS (silent corruption on torch 2.14.0)
            out[k] = torch.stack([got[l][0, -1].cpu().float() for l in layers])
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(texts)}", flush=True)
    for h in hooks:
        h.remove()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="build output dir (stats.pt, holdout.pt)")
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/natural_text")
    ap.add_argument("--model", default=SUBJECT_MODEL)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--per-word", type=int, default=20)
    ap.add_argument("--context-words", type=int, default=60)
    ap.add_argument("--min-words", type=int, default=8)
    ap.add_argument("--layers", default="16,20,24,28,31")
    ap.add_argument("--lambdas", default="0.01,0.1,1,10")
    args = ap.parse_args()
    out = args.out or args.run / "natural_text"
    out.mkdir(parents=True, exist_ok=True)

    st = torch.load(args.run / "stats.pt")
    held = torch.load(args.run / "holdout.pt")
    keys = st["keys"]
    counts = st["half_counts"].double()
    live = counts.sum(1) > 0
    vocab = [json.loads(line) for line in open(args.vocab, encoding="utf-8")]

    # ---- natural-text contexts and residuals (cached) -------------------------
    ctx_path, res_path = out / "contexts.jsonl", out / "residuals.pt"
    if ctx_path.exists() and res_path.exists():
        ctx = pd.read_json(ctx_path, lines=True)
        H_nat = torch.load(res_path).float()
    else:
        ctx = find_contexts(vocab, {k for k, a in zip(keys, live) if a}, args.per_word,
                            args.context_words, args.min_words)
        print(f"{len(ctx)} natural contexts for {ctx.key.nunique()} words")
        ctx.to_json(ctx_path, orient="records", lines=True, force_ascii=False)
        H_nat = read_residuals(ctx.text.tolist(), args.model, args.device)
        torch.save(H_nat.half(), res_path)

    widx = {k: i for i, k in enumerate(keys)}
    sets = {"natural": (H_nat.double(), torch.tensor([widx[k] for k in ctx.key])),
            "generated": (held["residuals"].double(), torch.tensor([widx[k] for k in held["keys"]]))}
    # compare on the same words
    nat_words = set(ctx.key)
    gen_mask = torch.tensor([k in nat_words for k in held["keys"]])
    sets["generated"] = (sets["generated"][0][gen_mask], sets["generated"][1][gen_mask])

    # ---- score against templates refit at each ridge --------------------------
    sums, mu, cov = st["half_sums"].double(), st["mean"].double(), st["cov"].double()
    d = cov.shape[-1]
    m_w = sums.sum(1) / counts.sum(1).clamp(min=1).view(-1, 1, 1)          # [W, L, d]
    summary, per_passage = [], {}
    for l in [int(x) for x in args.layers.split(",")]:
        for lam_rel in [float(x) for x in args.lambdas.split(",")]:
            lam = lam_rel * torch.trace(cov[l]) / d
            chol = torch.linalg.cholesky(cov[l] + lam * torch.eye(d, dtype=torch.float64))
            T = torch.cholesky_solve((m_w[:, l] - mu[l]).T, chol).T           # [W, d]
            for name, (H, own) in sets.items():
                sc = (H[:, l] - mu[l]) @ T.T
                sc[:, ~live] = float("-inf")
                rank = (sc > sc[torch.arange(len(own)), own][:, None]).sum(1)
                summary.append({"set": name, "layer": l, "lambda_rel": lam_rel, "n": len(own),
                                "top1": float((rank == 0).float().mean()),
                                "top5": float((rank < 5).float().mean()),
                                "median_rank": float(rank.float().median())})
                per_passage[(name, l, lam_rel)] = (rank, sc.argmax(1))

    s = pd.DataFrame(summary)
    s.to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\nlive words: {int(live.sum())} (chance top-1 {1 / int(live.sum()):.3f})")
    print(s.pivot_table(index=["layer", "lambda_rel"], columns="set",
                        values=["top1", "top5", "median_rank"]).round(3).to_string())

    best = s[s.set == "natural"].sort_values("top1").iloc[-1]
    l, lam_rel = int(best.layer), best.lambda_rel
    rank, pred = per_passage[("natural", l, lam_rel)]
    ctx["rank"], ctx["predicted"] = rank.numpy(), [keys[i] for i in pred]
    ctx.to_json(out / f"contexts_scored_L{l}_lam{lam_rel:g}.jsonl", orient="records", lines=True, force_ascii=False)
    g_rank, _ = per_passage[("generated", l, lam_rel)]
    gen_keys = [k for k, m in zip(held["keys"], gen_mask.tolist()) if m]
    per_word = (ctx.groupby("key").agg(n_natural=("rank", "size"), top1_natural=("rank", lambda r: (r == 0).mean()))
                .join(pd.DataFrame({"key": gen_keys, "r": g_rank.numpy()}).groupby("key")
                      .agg(top1_generated=("r", lambda r: (r == 0).mean()))))
    per_word.to_csv(out / "per_word.csv")
    print(f"\nat layer {l}, lambda_rel {lam_rel:g}:")
    print("natural top-1 by whether the word already appears in the context:")
    print(ctx.groupby("word_in_context").agg(n=("rank", "size"), top1=("rank", lambda r: (r == 0).mean())).round(3).to_string())
    print("\nmost frequent natural-text predictions (hub check):")
    print(ctx.predicted.value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
