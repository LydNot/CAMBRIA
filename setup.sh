#!/usr/bin/env bash
# ARENA per-session setup.  Usage:  source /workspace/setup.sh
# Lives on /workspace, so the venv + this script survive stop/start AND migrations.
# torch/CUDA come from the base image (which also survives migrations).

VENV=/workspace/arena-venv
HF=/workspace/.cache/huggingface

if [ ! -d "$VENV" ]; then
    echo ">> First run: building venv (~1-2 min, one time only)..."
    python3 -m venv --system-site-packages "$VENV"
    source "$VENV/bin/activate"
    pip install -q -U pip
    # ARENA pins (transformers<5, pandas<3); numpy<2 to stay compatible with image torch.
    # --ignore-installed forces these INTO the venv so they persist on the volume.
    pip install --ignore-installed \
        "transformers<5" "pandas<3" "numpy<2" \
        peft plotly python-dotenv jaxtyping einops tqdm ipywidgets ipykernel pytest tabulate
    # tie HF_HOME to the venv so Cursor's kernel inherits it on activation
    echo "export HF_HOME=$HF" >> "$VENV/bin/activate"
    echo ">> Build complete."
else
    source "$VENV/bin/activate"
fi

export HF_HOME=$HF
echo ">> Ready. HF_HOME=$HF"
echo ">> Set Cursor interpreter to: $VENV/bin/python"
python -c "import transformers,sys; print('   transformers', transformers.__version__, '| python', sys.executable)"
