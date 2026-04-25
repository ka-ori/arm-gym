#!/usr/bin/env bash
# Reference launcher for Hugging Face Jobs: MVP (50-step) training on a10g-large (or similar).
# Requires: HF_TOKEN in env for adapter upload. Do not commit tokens.
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN}"
echo "=== System deps ==="
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  qemu-user-static binutils-aarch64-linux-gnu gcc-aarch64-linux-gnu \
  libc6-dev-arm64-cross ca-certificates curl gnupg lsb-release >/dev/null
CODENAME="${CODENAME:-$(lsb_release -cs 2>/dev/null || echo bookworm)}"
curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm.gpg
echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/${CODENAME}/ llvm-toolchain-${CODENAME}-21 main" \
  > /etc/apt/sources.list.d/llvm21.list
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq clang-21 llvm-21 llvm-21-tools >/dev/null \
  || apt-get install -y -qq clang llvm llvm-tools
export PATH="/usr/lib/llvm-21/bin:${PATH}"
echo "=== Python deps ==="
python3 -m pip install -q --upgrade pip
python3 -m pip install -q --no-input \
  'trl==0.20.0' 'transformers>=4.55,<4.58' 'accelerate>=1.0' 'peft>=0.13' 'datasets>=3.0' \
  'bitsandbytes>=0.45' 'torch>=2.3' 'numpy>=1.26' 'pydantic>=2.7' sentencepiece protobuf \
  huggingface_hub
V6_URL="https://huggingface.co/datasets/ZDC-M01/arm-gym-pkg/resolve/main/v6_train.py"
WHEEL_URL="https://huggingface.co/datasets/ZDC-M01/arm-gym-pkg/resolve/main/arm_gym-0.1.0-py3-none-any.whl"
python3 -m pip install -q --force-reinstall "arm_gym @ ${WHEEL_URL}"
python3 -m pip uninstall -y torchao vllm unsloth unsloth_zoo xformers 2>/dev/null || true
echo "=== Run GRPO (MVP profile = 50 steps via hf/PROFILE) ==="
curl -sSL "$V6_URL" -o /tmp/v6_train.py
cd /tmp
python3 v6_train.py
