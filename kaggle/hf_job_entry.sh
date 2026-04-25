#!/usr/bin/env bash
# Entry point for a Hugging Face "Job" (cloud GPU) run: clone Space → system deps → train.
# Use this as the job **Command** (see kaggle/HF_JOB.md) or: bash kaggle/hf_job_entry.sh
#
# Environment (set in the Job UI "Secrets" / env):
#   HF_TOKEN              — read token, for git clone of the Space
#   HF_SPACE              — optional, default: kaori02/arm-gym  (user/repo)
#   HF_RUN_SETUP          — optional, default: 1  (0 skips kaggle/setup.sh)
#   HF_TRAIN_EXTRAS       — optional extra args to kaggle/train.py
#   HF_CLONE_DIR          — optional, override checkout directory name

set -euo pipefail

if [ -n "${HF_SPACE:-}" ]; then
  HF_SPACE_USER="${HF_SPACE%%/*}"
  HF_SPACE_REPO="${HF_SPACE#*/}"
else
  HF_SPACE_USER="${HF_SPACE_USER:-kaori02}"
  HF_SPACE_REPO="${HF_SPACE_REPO:-arm-gym}"
fi

REPO_NAME="${REPO_NAME:-arm-gym-work}"
CHECKOUT="${HF_CLONE_DIR:-${PWD}/${REPO_NAME}}"

if [ ! -f kaggle/train.py ]; then
  : "${HF_TOKEN:?set HF_TOKEN in Job secrets to clone the Space}"
  if [ -d "${CHECKOUT}/.git" ]; then
    echo "[hf_job_entry] updating ${CHECKOUT}"
    git -C "${CHECKOUT}" pull --ff-only
  else
    echo "[hf_job_entry] cloning ${HF_SPACE_USER}/${HF_SPACE_REPO} -> ${CHECKOUT}"
    git clone --depth 1 \
      "https://user:${HF_TOKEN}@huggingface.co/spaces/${HF_SPACE_USER}/${HF_SPACE_REPO}" \
      "${CHECKOUT}"
  fi
  cd "${CHECKOUT}"
else
  echo "[hf_job_entry] kaggle/train.py in cwd, using ${PWD} as repo root"
fi

if [ "${HF_RUN_SETUP:-1}" = "1" ] && [ -f kaggle/setup.sh ]; then
  echo "[hf_job_entry] kaggle/setup.sh (toolchain + llvm-mca; needs root in container)"
  bash kaggle/setup.sh
fi

export HF_SKIP_CLONE=1
export HF_CLONE_DIR="${PWD}"
# Explicit default so logs show the fix even if an older config.json is used.
export HF_TRAIN_EXTRAS="${HF_TRAIN_EXTRAS:---assembly-min-new-tokens-before-close 64}"

exec bash kaggle/hf_train_job.sh
