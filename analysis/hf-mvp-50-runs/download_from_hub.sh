#!/usr/bin/env bash
# Download full model-repo snapshot. Requires: uv (recommended) or huggingface_hub in python3.
# Usage: HF_TOKEN=... ./download_from_hub.sh [ZDC-M01/arm-gym-mvp-50] [./out-dir]
set -euo pipefail
export REPO_ARG="${1:-ZDC-M01/arm-gym-mvp-50}"
export OUT_ARG="${2:-./arm-gym-mvp-50-snapshot}"
uv run --with huggingface_hub python -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=os.environ['REPO_ARG'],
    local_dir=os.environ['OUT_ARG'],
    token=os.environ.get('HF_TOKEN'),
    repo_type='model',
)
print('OK', os.environ['OUT_ARG'])
"
