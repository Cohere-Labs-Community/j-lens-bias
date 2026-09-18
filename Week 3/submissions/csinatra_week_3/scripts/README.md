# Template lens for BBQ

Builds template-lens vectors (workspace paper, A.9.1) for Qwen3.5-4B over a
vocabulary of BBQ group surfaces, HolisticBias descriptors and nonce controls.
Heavy steps run on a cloud H100; the BBQ readout and analysis run locally.

| file | where | does |
|---|---|---|
| `build_vocab.py` | local | vocabulary + fixed pilot subset → `vocab.jsonl` |
| `generate_passages.py` | cloud, `gen` env | Qwen3.6-27B writes passages ending just before each word |
| `build_templates.py` | cloud, `build` env | residuals → sufficient statistics, templates, held-out readout |
| `parity_check.py` | local | CUDA vs MPS residuals on the held-out passages |
| `setup_cloud.sh`, `run_cloud.sh` | cloud | environments and the pipeline |

## Run

```bash
# local
python build_vocab.py --bbq "../../../../Week 2/_bbq" --out ../../../_template_lens
rsync -av --exclude __pycache__ ./ ubuntu@HOST:~/tl/
rsync -av ../../../_template_lens/vocab.jsonl ubuntu@HOST:~/tl/data/

# cloud
export HF_TOKEN=...
bash setup_cloud.sh                      # also enables lingering, so runs survive logout
systemd-run --user --scope --unit=tl tmux new-session -d -s tl -e HF_TOKEN="$HF_TOKEN" \
    "cd ~/tl && bash run_cloud.sh full 2>&1 | tee ~/tl/session_full.log; exec bash"

# watching, from anywhere -- read-only: a stray Ctrl-C in an attached pane kills the run
ssh -t ubuntu@HOST "tmux attach -r -t tl"
ssh -t ubuntu@HOST "less -R +F ~/tl/session_full.log"

# local
rsync -av ubuntu@HOST:~/tl/out_full ../../../_template_lens/
python parity_check.py --holdout ../../../_template_lens/out_full/holdout.pt
python natural_text_check.py --run ../../../_template_lens/out_full --vocab ../../../_template_lens/vocab.jsonl
```

## Environment

Both cloud envs use Python 3.14. `build` pins torch 2.14.0 and transformers
5.16.1 to match local, because Qwen3.5's modeling code and the MPS backend both
change between releases. `gen` is separate because vLLM 0.29.0 pins torch
2.13.0; it only produces text.

On MPS, never write `.to(device, dtype)`: on torch 2.14.0 `.to("cpu",
torch.float64)` from a bf16 tensor silently returns wrong values. Move and
convert in two calls.

## Pilot go/no-go

Decided before the pilot runs. The pilot is 12 fixed words, one of each kind the
pipeline has to handle (see `build_vocab.py`).

| check | go if | source |
|---|---|---|
| setup | both envs install; `fla` resolves the delta rule; vLLM + Qwen3.6-27B load | `setup_cloud.sh` |
| cost | projected full run ≤ 2× the ~$12–13 estimate | `generation_log.json` tokens/s, `build_seconds` |
| parsing | JSON parse failures < 10% of requests | `generation_log.json` |
| yield | ≥ 60 passages kept per word; no source rejecting > 40% | `build_summary.json` verdicts by source |
| plausibility | recorded only, never filtered: a cutoff would keep only contexts where the subject model already expects the word | `phrase_mean_logprob_by_source` |
| screen | writer-model judge (slot fit, no stereotype cue) rejects < 60% overall; group terms alone pass ≥ 60% | `generation_log.json` `judge_*` |
| held-out readout | late-band top-1 well above chance (1/12 ≈ 8%) | `holdout_readout` |
| stability | split-half cosine positive and rising with depth | `split_half_cosine_median_by_layer` |
| parity | workspace band min cosine ≥ 0.999; median rel err ≤ 5% | `parity_check.py` (MPS-vs-MPS baseline: 0.9%, cos 0.9999) |
| quality | read ~20 passages across sources for fluency and stereotyped framing | `passage_verdicts.jsonl` |

Held-out readout and split-half are uninformative at smoke-test scale (2–3
passages per half); they are only meaningful from the pilot on.
