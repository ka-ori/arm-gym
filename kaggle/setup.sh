#!/usr/bin/env bash
# Kaggle bootstrap — run as the first notebook cell via: !bash kaggle/setup.sh
# Installs aarch64 toolchain + QEMU + LLVM 21 + training stack.
set -euo pipefail

if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "[setup] error: need python3 or python on PATH" >&2
  exit 1
fi

echo "[setup] apt toolchain"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  qemu-user-static \
  binutils-aarch64-linux-gnu \
  gcc-aarch64-linux-gnu \
  ca-certificates curl gnupg lsb-release >/dev/null

echo "[setup] llvm-21 from apt.llvm.org"
curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
  | gpg --dearmor -o /usr/share/keyrings/llvm.gpg
CODENAME=$(lsb_release -cs)
echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/${CODENAME}/ llvm-toolchain-${CODENAME}-21 main" \
  > /etc/apt/sources.list.d/llvm21.list
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  clang-21 llvm-21 llvm-21-tools >/dev/null || {
    echo "[setup] llvm-21 unavailable, falling back to distro llvm"
    apt-get install -y -qq clang llvm llvm-tools
  }

export PATH=/usr/lib/llvm-21/bin:$PATH

echo "[setup] python deps  (${PYTHON})"
"${PYTHON}" -m pip install -q --upgrade pip
"${PYTHON}" -m pip install -q -e ".[train,dev]" || "${PYTHON}" -m pip install -q \
  torch "trl==0.20.0" "transformers>=4.55,<4.58" "accelerate>=1.0" "peft>=0.13" \
  "datasets>=3.0" "bitsandbytes>=0.45" pydantic fastapi uvicorn numpy httpx

echo "[setup] optional: unsloth + vllm (multi-GPU fragility; wrapped in || true)"
"${PYTHON}" -m pip install -q "unsloth>=2024.10" 2>/dev/null || echo "[setup] unsloth skipped"
"${PYTHON}" -m pip install -q "vllm>=0.6"         2>/dev/null || echo "[setup] vllm skipped"

echo "[setup] toolchain verify"
which aarch64-linux-gnu-gcc aarch64-linux-gnu-as qemu-aarch64-static || true
which clang-21 llvm-mca-21 2>/dev/null || which clang llvm-mca
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"${PYTHON}" - <<'PY'
from arm_gym.compile_baseline import detect_toolchain
tc = detect_toolchain()
print(f"[setup] tc.clang={tc.clang} tc.gcc_aarch64={tc.gcc_aarch64} "
      f"tc.mca={tc.mca} tc.mcpu={tc.mcpu} disclosed={tc.mcpu_disclosed}")
PY
echo "[setup] done"
