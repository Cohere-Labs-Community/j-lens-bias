"""Build the template-lens vocabulary. Runs locally; no model needed.

Primary: every distinct group surface in BBQ's label-type items, merged across
spelling variants. Secondary: HolisticBias descriptors for the axes those BBQ
categories map to, for coverage beyond BBQ's own phrasing. Control: HolisticBias
nonce words, which name no real group and should show no bias.

Writes vocab.jsonl (one entry per template) to --out. The pilot subset is fixed
here rather than chosen at run time, so the cloud pilot is reproducible.

    python build_vocab.py --bbq "../../Week 2/_bbq" --out ../../_template_lens
"""

import argparse
import collections
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from common import SUBJECT_MODEL, vocab_key

BBQ_CATEGORIES = ["Age", "Disability_status", "Gender_identity", "Nationality",
                  "Physical_appearance", "Race_ethnicity", "Race_x_SES",
                  "Race_x_gender", "Religion", "SES", "Sexual_orientation"]

# BBQ category -> HolisticBias axis. The two intersectional BBQ categories are
# covered by their constituent axes.
HB_AXIS = {"Age": "age", "Disability_status": "ability",
           "Gender_identity": "gender_and_sex", "Nationality": "nationality",
           "Physical_appearance": "body_type", "Race_ethnicity": "race_ethnicity",
           "Religion": "religion", "SES": "socioeconomic_class",
           "Sexual_orientation": "sexual_orientation"}
INTERSECTIONAL = {"Race_x_SES", "Race_x_gender"}

HB_URL = ("https://raw.githubusercontent.com/facebookresearch/ResponsibleNLP/"
          "main/holistic_bias/dataset/v1.1/descriptors.json")

# One of each kind the pipeline has to handle, so a pilot surfaces every failure
# mode before the full run. First match wins; a missing entry is reported.
PILOT_WANTS = [
    ("bbq single token", lambda e: e["source"] == "bbq" and e["word"] == "Muslim"),
    ("bbq multi-token nationality", lambda e: e["source"] == "bbq"
     and "Nationality" in e["categories"] and e["n_tokens"] > 1),
    ("bbq numeric age", lambda e: e["source"] == "bbq" and e["word"] == "78-year-old"),
    ("bbq occupation (SES)", lambda e: e["source"] == "bbq" and e["word"] == "janitor"),
    ("bbq disability phrase", lambda e: e["source"] == "bbq"
     and "Disability_status" in e["categories"] and e["n_tokens"] >= 3),
    ("bbq Race_x_SES phrase", lambda e: e["source"] == "bbq"
     and "Race_x_SES" in e["categories"]),
    ("bbq Race_x_gender phrase", lambda e: e["source"] == "bbq"
     and "Race_x_gender" in e["categories"]),
    ("holisticbias reviewed", lambda e: e["source"] == "holisticbias"
     and e["hb_preference"] == "reviewed"),
    ("holisticbias dispreferred", lambda e: e["source"] == "holisticbias"
     and e["hb_preference"] == "dispreferred"),
    ("holisticbias polarizing", lambda e: e["source"] == "holisticbias"
     and e["hb_preference"] == "polarizing"),
    ("nonce control 1", lambda e: e["source"] == "nonce"),
    ("nonce control 2", lambda e: e["source"] == "nonce"),
]


def bbq_surfaces(bbq_dir: Path) -> dict[str, collections.Counter]:
    """key -> Counter of (surface, category) over BBQ label-type items."""
    frames = []
    for cat in BBQ_CATEGORIES:
        with open(bbq_dir / f"{cat}.jsonl", encoding="utf-8") as fh:
            frames.append(pd.DataFrame([json.loads(line) for line in fh]))
    bbq = pd.concat(frames, ignore_index=True)
    bbq["question_index"] = bbq.question_index.astype(int)
    meta = (pd.read_csv(bbq_dir / "additional_metadata.csv")
            .drop_duplicates(["category", "question_index", "example_id"]))
    items = bbq.merge(meta, on=["category", "question_index", "example_id"])
    items = items[items.target_loc.notna() & (items.label_type == "label")]

    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for info, a0, a1, a2, cat in zip(items.answer_info, items.ans0, items.ans1,
                                     items.ans2, items.category):
        for key, raw in (("ans0", a0), ("ans1", a1), ("ans2", a2)):
            surface, label = info[key]
            if label == "unknown":
                continue
            # 344 Race_x_gender items carry an empty answer_info surface; the
            # top-level ansN field has the text (see Week 2 section 1).
            surface = surface or re.sub(r"^[Tt]he\s+", "", raw).strip()
            if surface:
                out[vocab_key(surface)][(surface, cat)] += 1
    return out


def load_holisticbias(cache: Path) -> list[dict]:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(HB_URL, timeout=60) as r:
            cache.write_bytes(r.read())
    rows = []
    for axis, buckets in json.loads(cache.read_text()).items():
        for bucket, entries in buckets.items():
            for e in entries:
                word = e["descriptor"] if isinstance(e, dict) else e
                pref = e.get("preference", "") if isinstance(e, dict) else ""
                rows.append({"word": word, "axis": axis, "bucket": bucket,
                             "preference": pref})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbq", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
    n_tokens = lambda w: len(tok(" " + w, add_special_tokens=False).input_ids)

    entries: dict[str, dict] = {}

    for key, counts in bbq_surfaces(args.bbq).items():
        by_form = collections.Counter()
        for (surface, _), n in counts.items():
            by_form[surface] += n
        word = by_form.most_common(1)[0][0]
        entries[key] = {
            "key": key, "word": word, "source": "bbq",
            "variants": sorted(by_form),
            "categories": sorted({c for (_, c) in counts}),
            "n_bbq_items": int(sum(counts.values())),
            "hb_axis": None, "hb_bucket": None, "hb_preference": None,
        }

    wanted_axes = set(HB_AXIS.values()) | {"nonce"}
    for row in load_holisticbias(args.out / "holisticbias_descriptors.json"):
        if row["axis"] not in wanted_axes:
            continue
        key = vocab_key(row["word"])
        if key in entries:                  # BBQ already has it: annotate, don't duplicate
            entries[key].update(hb_axis=row["axis"], hb_bucket=row["bucket"],
                                hb_preference=row["preference"] or None)
            continue
        entries[key] = {
            "key": key, "word": row["word"],
            "source": "nonce" if row["axis"] == "nonce" else "holisticbias",
            "variants": [row["word"]], "categories": [], "n_bbq_items": 0,
            "hb_axis": row["axis"], "hb_bucket": row["bucket"],
            "hb_preference": row["preference"] or None,
        }

    # Intersectional surfaces ("Roma taxi driver") are read through their parts, so
    # each part gets a template rather than every combination getting one.
    known = set(entries)
    parts_of: dict[str, list[str]] = {}
    for key, e in list(entries.items()):
        words = key.split()
        if e["source"] != "bbq" or len(words) < 2 or not set(e["categories"]) <= INTERSECTIONAL:
            continue
        head = next((" ".join(words[:i]) for i in range(len(words) - 1, 0, -1)
                     if " ".join(words[:i]) in known), None)
        if head is None:
            continue
        rest = " ".join(words[len(head.split()):])
        parts_of[key] = [head, rest]
        del entries[key]
        if rest not in entries:
            rest_word = " ".join(re.split(r"[-\s]+", e["word"].strip())[len(head.split()):])
            entries[rest] = {"key": rest, "word": rest_word, "source": "bbq", "variants": [rest_word],
                             "categories": [], "n_bbq_items": 0,
                             "hb_axis": None, "hb_bucket": None, "hb_preference": None}
        for part in (head, rest):
            entries[part]["categories"] = sorted(set(entries[part]["categories"]) | set(e["categories"]))
            entries[part]["n_bbq_items"] += e["n_bbq_items"]
    assert all(p in entries for ps in parts_of.values() for p in ps)
    (args.out / "intersectional_parts.json").write_text(json.dumps(parts_of, indent=1))
    print(f"intersectional compounds split into parts: {len(parts_of)}")

    vocab = sorted(entries.values(), key=lambda e: (e["source"], e["key"]))
    for e in vocab:
        e["n_tokens"] = n_tokens(e["word"])
        e["pilot"] = None

    taken = set()
    for label, match in PILOT_WANTS:
        hit = next((e for e in vocab if e["key"] not in taken and match(e)), None)
        if hit is None:
            print(f"  pilot: no entry for {label!r}")
            continue
        hit["pilot"] = label
        taken.add(hit["key"])

    path = args.out / "vocab.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for e in vocab:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")

    df = pd.DataFrame(vocab)
    print(f"wrote {len(df)} entries -> {path}\n")
    print(df.groupby("source").agg(entries=("key", "size"),
                                   multi_token=("n_tokens", lambda s: int((s > 1).sum())))
          .to_string())
    print("\nBBQ entries by category (an entry can appear in several):")
    print(df[df.source == "bbq"].explode("categories").categories.value_counts().to_string())
    print(f"\nspelling variants merged: "
          f"{int((df.variants.str.len() > 1).sum())} entries had more than one form")
    print("\npilot subset:")
    print(df[df.pilot.notna()][["pilot", "word", "source", "n_tokens"]].to_string(index=False))


if __name__ == "__main__":
    main()
