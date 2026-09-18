#!/usr/bin/env bash
# Generate passages, then build templates, on the GPU box.
#
#   bash run_cloud.sh pilot                  # fixed 12-word pilot, ~15 min
#   bash run_cloud.sh full
#
# Generation and build run as separate processes in separate environments, so
# vLLM's GPU memory is fully released before the build loads the subject model.
#
# Detached, so it survives logout (setup_cloud.sh enables lingering):
#   systemd-run --user --scope --unit=tl tmux new-session -d -s tl \
#       "cd ~/tl && bash run_cloud.sh full 2>&1 | tee ~/tl/session_full.log; exec bash"
# Watch it read-only -- attaching read-write risks Ctrl-C killing the run:
#   ssh -t HOST "tmux attach -r -t tl"       # or: less -R +F ~/tl/session_full.log
set -euo pipefail
cd "$(dirname "$0")"

MODE=${1:?usage: run_cloud.sh pilot|full [build args...]}
shift || true
VOCAB=${VOCAB:-data/vocab.jsonl}   # uploaded here; override for a word subset
OUT=${OUT:-out_$MODE}

if [[ "$MODE" == "pilot" ]]; then
    GEN_FLAGS=(--pilot)
    BUILD_FLAGS=(--pilot --holdout-per-word 20)
elif [[ "$MODE" == "full" ]]; then
    GEN_FLAGS=()
    BUILD_FLAGS=(--holdout-per-word 5)
else
    echo "mode must be pilot or full" >&2; exit 1
fi

mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee "$OUT/gpu.txt"

# Each step gets its env's bin/ on PATH: vLLM and triton JIT-compile and shell
# out to tools (ninja) that live there.
echo "== generate ($MODE) =="
PATH="$PWD/envs/gen/bin:$PATH" envs/gen/bin/python generate_passages.py --vocab "$VOCAB" --out "$OUT" \
    "${GEN_FLAGS[@]}" 2>&1 | tee "$OUT/generate.log"

echo "== build templates ($MODE) =="
PATH="$PWD/envs/build/bin:$PATH" envs/build/bin/python build_templates.py --vocab "$VOCAB" \
    --passages "$OUT/passages" --out "$OUT" "${BUILD_FLAGS[@]}" "$@" \
    2>&1 | tee "$OUT/build.log"

"$HOME/.local/bin/uv" pip freeze --python envs/build/bin/python > "$OUT/build_env.txt"
echo "done -> $OUT"
