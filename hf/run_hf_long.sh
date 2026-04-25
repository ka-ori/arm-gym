#!/usr/bin/env bash
# 200-step Hugging Face Job launcher (same deps as run_hf_mvp.sh).
set -euo pipefail
export ARMGYM_PROFILE=long
DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "$DIR/run_hf_mvp.sh"
