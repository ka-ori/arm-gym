#!/usr/bin/env bash
# Force-push current branch to Hugging Face Space main, rewriting plot PNGs to Git LFS
# first so the Hub accepts the pack (raw PNG blobs are rejected).
#
#   git fetch origin && git checkout main && git pull
#   export HF_TOKEN=hf_...   # or use git credential for huggingface.co
#   ./scripts/sync_hf_space.sh
#
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

HF_USER="${HF_USER:-kaori02}"
HF_SPACE="${HF_SPACE:-arm-gym}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  REMOTE_URL="https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/${HF_USER}/${HF_SPACE}"
else
  REMOTE_URL="https://huggingface.co/spaces/${HF_USER}/${HF_SPACE}"
fi

REF=$(git symbolic-ref -q HEAD) || { echo "error: need a branch (detached HEAD not supported)" >&2; exit 1; }

git lfs install
git lfs migrate import --include="colab/results/plots/*.png" --include-ref="$REF"

echo "[sync_hf_space] pushing HEAD -> ${HF_USER}/${HF_SPACE}:main (force)"
git push --force "$REMOTE_URL" HEAD:main
