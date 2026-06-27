#!/usr/bin/env bash
# =============================================================================
# build_tl_venv.sh
# Builds a SEPARATE transformer_lens venv for ARENA sections 1.1, 1.2, and the
# classic mech-interp chapters. Does NOT touch /workspace/arena-venv (your AO /
# Chapter-4 env). Run once; it just creates the venv + registers its kernel.
#
# Why separate: transformer_lens pins transformers tightly and conflicts with the
# AO-tuned stack. Keeping the two families in separate venvs avoids the version
# war. Built the clean way (no --system-site-packages, torch+torchvision together)
# to avoid the torchvision/torch ABI mismatch that masquerades as a Bloom error.
#
# Usage:  bash /workspace/build_tl_venv.sh
# =============================================================================
set -e

TLVENV=/workspace/arena-tl-venv
HF=/workspace/.cache/huggingface          # shared model cache (same as AO venv)

if [ -d "$TLVENV" ]; then
    echo ">> $TLVENV already exists. Delete it first if you want a clean rebuild:"
    echo "   rm -rf $TLVENV"
    exit 0
fi

echo ">> Building transformer_lens venv at $TLVENV (one-time, ~several min for torch)..."
python3 -m venv "$TLVENV"                  # NOTE: no --system-site-packages (sealed)
source "$TLVENV/bin/activate"
pip install -U pip

# torch + torchvision installed TOGETHER => matched CUDA build (no nms mismatch).
# transformer_lens + the classic-interp stack. transformers is left to whatever
# transformer_lens wants (that's the whole point of isolating it here).
pip install torch torchvision \
    "transformer_lens>=2.16.1,<3.0.0" \
    transformers \
    einops jaxtyping \
    pandas numpy plotly \
    datasets \
    ipykernel ipywidgets tqdm pytest tabulate python-dotenv

# tie HF cache to the shared volume location so models persist & are shared with AO venv
echo "export HF_HOME=$HF" >> "$TLVENV/bin/activate"

# register the kernel under a DISTINCT name so you can tell it apart from "ARENA venv"
python -m ipykernel install --name arena-tl-venv --display-name "ARENA TL venv"

echo ">> Verifying..."
python -c "import torch, torchvision; print('torch', torch.__version__, '| cuda', torch.cuda.is_available()); import torchvision.ops; print('torchvision ops OK'); import transformer_lens; print('transformer_lens', transformer_lens.__version__); import transformers; print('transformers', transformers.__version__)"

deactivate
echo ">> Done. Kernel registered as 'ARENA TL venv'."
echo ">> Your AO venv (/workspace/arena-venv, 'ARENA venv') is untouched."
echo ">> When you reach 1.1/1.2: select the 'ARENA TL venv' kernel in Cursor."