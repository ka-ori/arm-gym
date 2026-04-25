#!/usr/bin/env bash
# One-shot runner for Hugging Face Jobs: pinned pip deps, consistent python -m pip,
# clone Space over HTTPS, then GRPO training with the "divergent" 50-step profile.
# For the same system toolchain (QEMU, LLVM, aarch64) as Kaggle, run kaggle/setup.sh
# in the job image or base Dockerfile before this script.
#
# === Environment ===
# HF_TOKEN
#   Read-only Hugging Face access token, embedded in the git clone URL as
#   https://user:${HF_TOKEN}@huggingface.co/...  (Hugging Face only — NOT GitHub).
# HF_SPACE_USER, HF_SPACE_REPO
#   Space path https://huggingface.co/spaces/<USER>/<REPO> (e.g. kaori02 arm-gym).
#   Alternatively set HF_SPACE=USER/REPO (e.g. HF_SPACE=kaori02/arm-gym).
# HF_CLONE_DIR
#   Where to clone the Space (default: ./hf_space_checkout).
# HF_SKIP_CLONE=1
#   Skip git clone; use HF_CLONE_DIR (or current directory) as the repo root.
#   Does not require HF_TOKEN or HF_SPACE_*.
# HF_TRAIN_EXTRAS
#   Optional extra args appended to kaggle/train.py (quoted string).

set -euo pipefail

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "[hf_train_job] error: need python3 or python on PATH" >&2
  exit 1
fi

echo "[hf_train_job] using: $(${PYTHON} -V 2>&1)"
"$PYTHON" -m pip install -q --upgrade pip
"$PYTHON" -m pip install -q --no-input \
  "trl==0.20.0" \
  "transformers>=4.55,<4.58" \
  "accelerate>=1.0" \
  "peft>=0.13" \
  "datasets>=3.0" \
  "bitsandbytes>=0.45" \
  "torch>=2.3" \
  "numpy>=1.26" \
  "pydantic>=2.7"

if [ -n "${HF_SPACE:-}" ]; then
  HF_SPACE_USER="${HF_SPACE%%/*}"
  HF_SPACE_REPO="${HF_SPACE#*/}"
fi

REPO_ROOT="${HF_CLONE_DIR:-./hf_space_checkout}"

if [ "${HF_SKIP_CLONE:-0}" = "1" ]; then
  if [ -n "${HF_CLONE_DIR:-}" ]; then
    REPO_ROOT="${HF_CLONE_DIR}"
  else
    REPO_ROOT="${PWD}"
  fi
  echo "[hf_train_job] HF_SKIP_CLONE=1  REPO_ROOT=${REPO_ROOT}"
else
  : "${HF_TOKEN:?[hf_train_job] set HF_TOKEN (Hugging Face token for clone; not GitHub)}"
  : "${HF_SPACE_USER:?[hf_train_job] set HF_SPACE_USER or HF_SPACE=USER/REPO}"
  : "${HF_SPACE_REPO:?[hf_train_job] set HF_SPACE_REPO or HF_SPACE=USER/REPO}"
  CLONE_URL="https://user:${HF_TOKEN}@huggingface.co/spaces/${HF_SPACE_USER}/${HF_SPACE_REPO}"
  if [ -d "${REPO_ROOT}/.git" ]; then
    echo "[hf_train_job] updating existing clone: ${REPO_ROOT}"
    git -C "${REPO_ROOT}" pull --ff-only
  else
    echo "[hf_train_job] cloning ${HF_SPACE_USER}/${HF_SPACE_REPO} -> ${REPO_ROOT}"
    git clone --depth 1 "${CLONE_URL}" "${REPO_ROOT}"
  fi
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
echo "[hf_train_job] PYTHONPATH=${PYTHONPATH}"

# Model: default 7B (SuperCoder / hackathon wiki). Use HF_MODEL for 14B/32B.
: "${HF_MODEL:=Qwen/Qwen2.5-Coder-7B-Instruct}"
# Divergent 50-step profile: LoRA 32, num_generations 4, max_train 128
TRAIN_CMD=(
  "${PYTHON}" kaggle/train.py
  --stack auto
  --model "${HF_MODEL}"
  --steps 50
  --lora-rank 32
  --num-generations 4
  --max-train 128
  --out runs/grpo-kaggle
)
if [ -n "${HF_TRAIN_EXTRAS:-}" ]; then
  # shellcheck disable=SC2206
  TRAIN_CMD+=(${HF_TRAIN_EXTRAS})
fi

echo "[hf_train_job] ${TRAIN_CMD[*]}"
exec "${TRAIN_CMD[@]}"
