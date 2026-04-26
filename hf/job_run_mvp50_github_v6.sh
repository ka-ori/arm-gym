#!/usr/bin/env bash
# Used by HF Jobs: run_hf_mvp.sh + v6 from GitHub main (latest harness/G=4/export).
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN}"
curl -sSL "https://raw.githubusercontent.com/ka-ori/arm-gym/main/hf/run_hf_mvp.sh" -o /tmp/mvp.sh
export ARMGYM_PROFILE=mvp
sed -i 's|^V6_URL=.*|V6_URL="https://raw.githubusercontent.com/ka-ori/arm-gym/main/hf/v6_train.py"|' /tmp/mvp.sh
exec bash /tmp/mvp.sh
