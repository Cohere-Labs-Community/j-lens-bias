#!/usr/bin/env bash
# One-time setup on the GPU box (Lambda H100, Ubuntu, CUDA toolkit preinstalled).
#
# Two environments, both Python 3.14:
#   envs/gen    vLLM, which pins its own torch. Only writes text passages, so
#               its torch version cannot reach anything downstream.
#   envs/build  torch and transformers pinned to the LOCAL versions, so the
#               residual stream the templates are fitted on matches the one the
#               local notebook reads.
#
#   export HF_TOKEN=...        # avoids hub rate limits
#   bash setup_cloud.sh
set -euo pipefail
cd "$(dirname "$0")"

TORCH=2.14.0
TRANSFORMERS=5.16.1
VLLM=0.29.0
JLENS_COMMIT=581d398613e5602a5af361e1c34d3a92ea82ba8e   # identical to the local package

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.14

# Long runs must outlive the SSH session that starts them: without lingering,
# logind tears down the session scope on logout and kills tmux with it.
sudo loginctl enable-linger "$USER" || echo "  (could not enable linger; keep an SSH session open)"

echo "== gen env (vLLM $VLLM) =="
uv venv envs/gen --python 3.14 --allow-existing
uv pip install --python envs/gen/bin/python "vllm==$VLLM"

echo "== build env (torch $TORCH, transformers $TRANSFORMERS) =="
uv venv envs/build --python 3.14 --allow-existing
uv pip install --python envs/build/bin/python \
    "torch==$TORCH" "transformers==$TRANSFORMERS" pandas numpy \
    flash-linear-attention \
    "jlens @ git+https://github.com/anthropics/jacobian-lens@$JLENS_COMMIT"
# No causal-conv1d: no wheel for torch 2.14/cp314, and torch's cu130 build does
# not match the image's CUDA 12.8 nvcc. transformers falls back per function, so
# the conv runs as F.conv1d (as on MPS) while fla still runs the delta rule.

echo "== version check =="
envs/build/bin/python - <<'PY'
import sys, torch, transformers
print("python", sys.version.split()[0], "| torch", torch.__version__,
      "| transformers", transformers.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
import fla  # noqa: F401  -- fail here, not mid-build
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
def impl(f):  # the implementation the kernel-fallback wrapper resolved to
    cells = dict(zip(f.__code__.co_freevars, f.__closure__ or ()))
    return cells["implementation"].cell_contents.__module__ if "implementation" in cells else "?"
for name in ["torch_chunk_gated_delta_rule", "causal_conv1d_fn"]:
    print(f"{name} -> {impl(getattr(m, name))}")
PY
envs/gen/bin/python -c "import vllm, sys; print('vllm', vllm.__version__, '| python', sys.version.split()[0])"
