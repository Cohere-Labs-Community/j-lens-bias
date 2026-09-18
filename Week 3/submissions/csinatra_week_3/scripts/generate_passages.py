"""Generate template-lens passages with the writer model. Runs on the GPU box
in the `gen` env (vLLM).

Follows the template-lens recipe (workspace paper, A.9.1): for each word w,
short passages written so that w is their natural continuation, ending just
before w, varying topic, frame and register, never using w itself. Output is
text only, which is why this step can live in its own environment.

    python generate_passages.py --vocab vocab.jsonl --out out --pilot
    python generate_passages.py --vocab vocab.jsonl --out out

Resumable: a word with a .done marker is skipped.
"""

import argparse
import hashlib
import json
import math
import re
import time
from pathlib import Path

from common import WRITER_MODEL, slug

REGISTERS = ["a news report", "casual conversation", "an academic paper",
             "literary fiction", "a social media post", "an instruction manual",
             "an interview transcript", "an official letter", "a personal diary",
             "a product or service review", "a textbook", "a podcast transcript"]
TOPICS = ["work", "school", "family life", "travel", "health care", "sports",
          "local politics", "cooking", "shopping", "technology", "housing",
          "a community event", "a courtroom", "a hospital", "public transport",
          "a job interview", "a wedding", "a neighborhood", "a museum visit",
          "volunteering"]

SYSTEM = "You write short, natural English passages for a language-model research study."

USER = """Write {k} different short passages. Each is one to three sentences, 20 to 80 words, and stops mid-sentence at exactly the point where the next word would naturally be "{word}".

Rules:
- Never use "{word}", or any word from it, anywhere in a passage.
- End each passage immediately before "{word}" would appear: no trailing punctuation, quotation mark or space.
- A reader should be able to guess "{word}" as the next word, or at least a word of the same kind. An ending that any of thousands of unrelated words could follow does not count.
- "{word}" should follow from the grammar and the situation, not from stereotypes or assumptions about any group of people.
- Write this batch as {register}. Where it fits, set the passages in {topic}; otherwise choose any everyday setting.
- Make every passage different in wording and situation, and at least 20 words long.

Example for the word "umbrella":
- Good: "The forecast promised heavy rain all afternoon, so before leaving the office she checked her bag twice to make sure she had packed her"
- Bad (too generic, almost any noun fits): "She walked into the kitchen and picked up her"

Return only a JSON array of {k} strings."""

JUDGE = """Passage: "{text}"
Next word: "{word}"

Answer two questions about this passage.
fits: Is "{word}" a natural next word here, and does the passage point toward it or toward a word of the same kind? Answer false if the ending is so generic that thousands of unrelated words would fit equally well.
stereotype: Does the passage lead to "{word}" through stereotypes or assumptions about a group of people (appearance, jobs, poverty, crime, behavior, ability), rather than through neutral facts in the text?

Return only JSON: {{"fits": true or false, "stereotype": true or false}}"""


def parse_judge(text: str) -> tuple[bool | None, bool | None]:
    m = re.search(r"\{.*?\}", text, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        j = {}
    fits, stereo = j.get("fits"), j.get("stereotype")
    return (fits if isinstance(fits, bool) else None, stereo if isinstance(stereo, bool) else None)


def variety(key: str, request_idx: int) -> tuple[str, str]:
    """Deterministic register and topic for one request."""
    h = int(hashlib.md5(f"{key}:{request_idx}".encode()).hexdigest(), 16)
    return REGISTERS[h % len(REGISTERS)], TOPICS[(h // len(REGISTERS)) % len(TOPICS)]


def parse_passages(text: str) -> tuple[list[str], bool]:
    """JSON array of strings; falls back to pulling quoted strings out."""
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            arr = json.loads(text[start:end + 1])
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                return [x.strip() for x in arr if x.strip()], True
        except json.JSONDecodeError:
            pass
    loose = []
    for s in re.findall(r'"((?:[^"\\]|\\.){20,})"', text):
        try:
            loose.append(json.loads(f'"{s}"', strict=False).strip())
        except json.JSONDecodeError:
            continue
    return loose, False


def build_requests(vocab: list[dict], raw_per_word: int, per_request: int) -> list[dict]:
    reqs = []
    n_req = math.ceil(raw_per_word / per_request)
    for e in vocab:
        for i in range(n_req):
            register, topic = variety(e["key"], i)
            reqs.append({"key": e["key"], "word": e["word"], "request_idx": i,
                         "register": register, "topic": topic,
                         "messages": [
                             {"role": "system", "content": SYSTEM},
                             {"role": "user", "content": USER.format(
                                 k=per_request, word=e["word"],
                                 register=register, topic=topic)}]})
    return reqs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pilot", action="store_true", help="only the fixed pilot subset")
    ap.add_argument("--raw-per-word", type=int, default=150,
                    help="passages requested per word, before filtering (target ~100 kept)")
    ap.add_argument("--per-request", type=int, default=10)
    ap.add_argument("--model", default=WRITER_MODEL)
    ap.add_argument("--chunk-words", type=int, default=100)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    # Qwen3.6's linear-attention layers need one state block per running sequence;
    # vLLM's default of 1024 exceeds the ~358 that fit on an H100 at 0.92.
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="print one prompt and exit, without loading vLLM")
    args = ap.parse_args()

    vocab = [json.loads(line) for line in open(args.vocab, encoding="utf-8")]
    if args.pilot:
        vocab = [e for e in vocab if e.get("pilot")]
    pdir = args.out / "passages"
    pdir.mkdir(parents=True, exist_ok=True)
    todo = [e for e in vocab if not (pdir / f"{slug(e['key'])}.done").exists()]
    print(f"{len(vocab)} words, {len(todo)} still to generate "
          f"({args.raw_per_word} raw passages each, {args.per_request} per request)")

    if args.dry_run:
        r = build_requests(todo[:1] or vocab[:1], args.raw_per_word, args.per_request)[0]
        print(json.dumps(r["messages"], indent=2))
        return
    if not todo:
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem, max_num_seqs=args.max_num_seqs, seed=args.seed,
              # prefix caching puts the hybrid model's state cache in 'align' mode, which
              # stalled on the 16k-prompt judge batch; shared prefixes here are tiny anyway
              enable_prefix_caching=False,
              limit_mm_per_prompt={"image": 0, "video": 0})

    log = {"model": args.model, "chunks": []}
    for c0 in range(0, len(todo), args.chunk_words):
        chunk = todo[c0:c0 + args.chunk_words]
        reqs = build_requests(chunk, args.raw_per_word, args.per_request)
        prompts = [tok.apply_chat_template(r["messages"], tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
                   for r in reqs]
        params = [SamplingParams(temperature=0.9, top_p=0.95,
                                 max_tokens=160 * args.per_request,
                                 seed=args.seed + 1000 * j)
                  for j in range(len(reqs))]
        t0 = time.monotonic()
        outs = llm.generate(prompts, params)
        dt = time.monotonic() - t0

        by_key: dict[str, list[dict]] = {}
        n_out_tokens = n_parse_fail = 0
        for r, o in zip(reqs, outs):
            comp = o.outputs[0]
            passages, ok = parse_passages(comp.text)
            n_out_tokens += len(comp.token_ids)
            n_parse_fail += not ok
            for pi, text in enumerate(passages):
                by_key.setdefault(r["key"], []).append({
                    "key": r["key"], "word": r["word"],
                    "request_idx": r["request_idx"], "passage_idx": pi,
                    "register": r["register"], "topic": r["topic"],
                    "text": text, "parse_ok": ok, "finish_reason": comp.finish_reason})

        # screen every passage with the writer model: slot fit and stereotype cues
        flat = [rec for v in by_key.values() for rec in v]
        judge_prompts = [tok.apply_chat_template(
            [{"role": "user", "content": JUDGE.format(text=rec["text"], word=rec["word"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False) for rec in flat]
        t1 = time.monotonic()
        judged = llm.generate(judge_prompts, SamplingParams(temperature=0.0, max_tokens=30))
        dt_judge = time.monotonic() - t1
        for rec, o in zip(flat, judged):
            fits, stereo = parse_judge(o.outputs[0].text)
            rec.update(judge_fits=fits, judge_stereotype=stereo,
                       judge_ok=fits is True and stereo is False)

        for e in chunk:
            s = slug(e["key"])
            with open(pdir / f"{s}.jsonl", "w", encoding="utf-8") as fh:
                for rec in by_key.get(e["key"], []):
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            (pdir / f"{s}.done").touch()

        stats = {"words": len(chunk), "requests": len(reqs), "seconds": round(dt, 1),
                 "output_tokens": n_out_tokens,
                 "tokens_per_second": round(n_out_tokens / max(dt, 1e-9)),
                 "parse_failures": n_parse_fail,
                 "passages": len(flat), "judge_seconds": round(dt_judge, 1),
                 "judge_unparsed": sum(r["judge_fits"] is None or r["judge_stereotype"] is None for r in flat),
                 "judge_fits": sum(r["judge_fits"] is True for r in flat),
                 "judge_stereotype": sum(r["judge_stereotype"] is True for r in flat),
                 "judge_ok": sum(r["judge_ok"] for r in flat)}
        log["chunks"].append(stats)
        print(f"  chunk {c0 // args.chunk_words + 1}: {stats}", flush=True)
        (args.out / "generation_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
