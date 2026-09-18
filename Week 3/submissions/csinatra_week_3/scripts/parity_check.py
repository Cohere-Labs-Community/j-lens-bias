"""Compare cloud (CUDA) and local (MPS) residuals on the same held-out passages.

Templates are fitted on H100 activations but applied to MPS activations, and
Qwen3.5's linear-attention layers run different kernels on each. This measures
how far the two residual streams actually differ, per layer, at the exact read
positions the templates use. Runs locally.

For calibration: MPS bfloat16 already differs from CPU float32 by ~1.7% in
relative norm on the same model (smoke test), so a few percent is expected.

    python parity_check.py --holdout ../../_template_lens/out_pilot/holdout.pt --n 48
"""

import argparse

import torch
import transformers

import jlens

from common import SUBJECT_MODEL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--model", default=SUBJECT_MODEL)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--n", type=int, default=48)
    args = ap.parse_args()

    held = torch.load(args.holdout)
    n = min(args.n, len(held["keys"]))
    cloud = held["residuals"][:n].float()                     # [N, L, d]

    dev = torch.device(args.device)
    hf = transformers.AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(dev)
    model = jlens.from_hf(hf, transformers.AutoTokenizer.from_pretrained(args.model))
    tok = model.tokenizer
    layers = list(range(model.n_layers))

    local = torch.empty_like(cloud)
    got: dict = {}
    hooks = [model.layers[l].register_forward_hook(
        lambda m, i, o, l=l: got.__setitem__(l, (o if torch.is_tensor(o) else o[0])))
        for l in layers]
    with torch.inference_mode():
        for k in range(n):
            ids = tok(held["texts"][k] + " " + held["phrases"][k], return_tensors="pt").input_ids
            got.clear()
            model._text_module(input_ids=ids.to(dev), use_cache=False)   # one at a time: no padding
            p = held["positions"][k]
            # two-step move: never .to(device, dtype) from MPS (silent corruption on torch 2.14.0)
            local[k] = torch.stack([got[l][0, p].cpu().float() for l in layers])
    for h in hooks:
        h.remove()

    rel = (local - cloud).norm(dim=-1) / cloud.norm(dim=-1)          # [N, L]
    cos = torch.nn.functional.cosine_similarity(local, cloud, dim=-1)
    print(f"{n} passages, {len(layers)} layers | non-finite local: "
          f"{int((~torch.isfinite(local)).sum())}")
    print(f"{'layer':>5} {'depth':>6} {'median rel err':>15} {'max rel err':>12} {'min cosine':>11}")
    for l in layers:
        print(f"{l:>5} {100 * l / (len(layers) - 1):>5.0f}% {float(rel[:, l].median()):>15.4f} "
              f"{float(rel[:, l].max()):>12.4f} {float(cos[:, l].min()):>11.5f}")
    print(f"\noverall: median rel err {float(rel.median()):.4f}, "
          f"worst {float(rel.max()):.4f}, min cosine {float(cos.min()):.5f}")


if __name__ == "__main__":
    main()
