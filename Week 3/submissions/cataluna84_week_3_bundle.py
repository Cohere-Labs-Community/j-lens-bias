"""Build a standalone input/ + output/ bundle for the Week 3 notebooks.

The notebooks read run artifacts from disk. The raw artifacts are ~185 MB,
which is too much to ship next to a submission, but almost all of that is
fields the notebooks never touch (Week 2's per-layer top-15 token lists alone
are 60 MB). This script projects each file down to the fields the notebooks
actually read and gzips the result, so the two notebooks become runnable from
a checkout or from Colab with no GPU and no access to my machine.

Usage:
    python analysis/bundle_week3.py \
        --wg-out analysis/out/week3/winogender \
        --steer-out analysis/out/week3/steering \
        --bbq-out analysis/out/bbq \
        --fu-out analysis/out/week3/followups \
        --data analysis/data \
        --target "/path/to/Week 3/submissions"
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from pathlib import Path

# Every field of the Winogender records is read by some cell, so that file is
# gzipped whole. The Week 2 BBQ records are the opposite case: the per-layer
# top-15 token lists are 60 of its 63 MB and no cell touches them.
BBQ_RECORD_KEYS = ("format", "category", "example_id", "polarity", "condition",
                   "pred_letter", "stereo_loc", "counter_loc", "unknown_loc")

# Small derived files that are copied verbatim.
WG_JSON = ("fidelity.json", "sanity.json", "bls_correlation.json",
           "occupation_lean.json", "lens_behaviour.json", "hedging.json")
ST_JSON = ("arms.json", "sham_pair.json", "validate.json", "vignettes.json",
           "summary.json")
# domdir.json holds the raw learned direction (2560 floats x 32 layers, 1.8 MB)
# and is an input to summarize, not to any notebook cell, so it stays out.
FU_JSON = ("followups_summary.json", "gensteer.json")
FU_PNG = ("fig_fullsent.png", "fig_bands.png", "fig_dom.png",
          "fig_frontier_seeds.png")


def _write_jsonl_gz(rows, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest, "wt", compresslevel=9) as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    return dest.stat().st_size


def _copy_jsonl_gz(src: Path, dest: Path) -> int:
    """Gzip a JSONL file unchanged (used where every field is read)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fin, gzip.open(dest, "wb", compresslevel=9) as fout:
        shutil.copyfileobj(fin, fout)
    return dest.stat().st_size


def project_bbq(src: Path):
    """Keep the ARC records, the eight scalars, and the per-layer ranks."""
    for line in src.open():
        r = json.loads(line)
        if r.get("format") != "arc":
            continue
        out = {k: r[k] for k in BBQ_RECORD_KEYS if k in r}
        out["layers"] = [{"layer": lr["layer"], "fr": lr["fr"]}
                         for lr in r["layers"]]
        yield out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wg-out", default="analysis/out/week3/winogender")
    ap.add_argument("--steer-out", default="analysis/out/week3/steering")
    ap.add_argument("--bbq-out", default="analysis/out/bbq")
    ap.add_argument("--fu-out", default="analysis/out/week3/followups")
    ap.add_argument("--data", default="analysis/data")
    ap.add_argument("--target", required=True,
                    help="directory to create input/ and output/ inside")
    args = ap.parse_args()

    wg, st = Path(args.wg_out), Path(args.steer_out)
    bq, fu, data = Path(args.bbq_out), Path(args.fu_out), Path(args.data)
    target = Path(args.target)
    inp, out = target / "input", target / "output"
    for d in (inp, out / "winogender", out / "steering", out / "followups",
              out / "bbq"):
        d.mkdir(parents=True, exist_ok=True)

    sizes: dict[str, int] = {}

    # --- inputs: the two public data files the harnesses read -------------
    for name in ("winogender_test.tsv", "occupations-stats.tsv"):
        src = data / name
        if src.exists():
            shutil.copy2(src, inp / name)
            sizes[f"input/{name}"] = (inp / name).stat().st_size

    # --- detection artifacts ---------------------------------------------
    sizes["output/winogender/records.jsonl.gz"] = _copy_jsonl_gz(
        wg / "records.jsonl", out / "winogender" / "records.jsonl.gz")
    for name in WG_JSON:
        if (wg / name).exists():
            shutil.copy2(wg / name, out / "winogender" / name)
            sizes[f"output/winogender/{name}"] = (
                out / "winogender" / name).stat().st_size

    # --- Week 2 carry-forward --------------------------------------------
    sizes["output/bbq/records.jsonl.gz"] = _write_jsonl_gz(
        project_bbq(bq / "records.jsonl"), out / "bbq" / "records.jsonl.gz")

    # --- steering artifacts ----------------------------------------------
    for name in ("steer_winogender.jsonl", "steer_bbq.jsonl", "fluency.jsonl"):
        if (st / name).exists():
            sizes[f"output/steering/{name}.gz"] = _copy_jsonl_gz(
                st / name, out / "steering" / f"{name}.gz")
    for name in ST_JSON:
        if (st / name).exists():
            shutil.copy2(st / name, out / "steering" / name)
            sizes[f"output/steering/{name}"] = (
                out / "steering" / name).stat().st_size

    # --- follow-up artifacts ---------------------------------------------
    for name in ("fullsent.jsonl", "bandsweep.jsonl", "randseeds.jsonl",
                 "domsteer.jsonl", "dommatch.jsonl"):
        if (fu / name).exists():
            sizes[f"output/followups/{name}.gz"] = _copy_jsonl_gz(
                fu / name, out / "followups" / f"{name}.gz")
    for name in FU_JSON + FU_PNG:
        if (fu / name).exists():
            shutil.copy2(fu / name, out / "followups" / name)
            sizes[f"output/followups/{name}"] = (
                out / "followups" / name).stat().st_size

    total = sum(sizes.values())
    for k in sorted(sizes, key=lambda k: -sizes[k]):
        print(f"  {sizes[k] / 1024:9.1f} KiB  {k}")
    print(f"bundle total: {total / 1024 / 1024:.2f} MiB in {target}")


if __name__ == "__main__":
    main()
