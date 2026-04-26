#!/usr/bin/env bash
# Upload hf/v6_train.py to the Jobs dataset. Requires HF_TOKEN (or huggingface-cli login).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
: "${HF_TOKEN:?Set HF_TOKEN (read-only is enough for upload if permitted)}"
export HF_TOKEN
uv run --with huggingface_hub python <<'PY'
from huggingface_hub import HfApi
from pathlib import Path
import os
p = Path("hf/v6_train.py")
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(
    path_or_fileobj=str(p),
    path_in_repo="v6_train.py",
    repo_id="ZDC-M01/arm-gym-pkg",
    repo_type="dataset",
    commit_message="chore: sync v6_train.py from arm-gym repo (Jobs curl target)",
)
print("OK: ZDC-M01/arm-gym-pkg  path=v6_train.py")
PY
