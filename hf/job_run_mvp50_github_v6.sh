#!/usr/bin/env bash
# Used by HF Jobs: run_hf_mvp + v6 from GitHub main. Uses Python fetch (no curl).
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN}"
BASE="https://raw.githubusercontent.com/ka-ori/arm-gym/main"
export ARMGYM_PROFILE=mvp
python3 -c "
import urllib.request, ssl
ctx = ssl.create_default_context()
u = '${BASE}/hf/run_hf_mvp.sh'
open('/tmp/mvp.sh', 'wb').write(urllib.request.urlopen(u, context=ctx).read())
print('fetched', u)
"
python3 -c "
p = '/tmp/mvp.sh'
c = open(p).read()
old = 'V6_URL=\"https://huggingface.co/datasets/ZDC-M01/arm-gym-pkg/resolve/main/v6_train.py\"'
new = 'V6_URL=\"${BASE}/hf/v6_train.py\"'
if old not in c:
    raise SystemExit('run_hf_mvp.sh: expected V6_URL line missing; rebase script')
open(p, 'w').write(c.replace(old, new, 1))
print('patched V6_URL -> main v6')
"
exec bash /tmp/mvp.sh
