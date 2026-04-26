#!/usr/bin/env bash
# One command to start a Hugging Face cloud Job.
# Prereqs: pip install -U "huggingface_hub[cli]"  &&  hf auth login
#          Jobs require HF Pro+ (or Team/Enterprise). Docs:
#          https://huggingface.co/docs/huggingface_hub/main/en/guides/jobs
#
#   ./kaggle/hf_run_job.sh                # default: a10g-large, Qwen2.5-Coder-7B, 3h
#   HF_FLAVOR=a100-large HF_MODEL=Qwen/...-14B  ./kaggle/hf_run_job.sh
#   HF_FLAVOR=h200 HF_MODEL=Qwen/...-32B  ./kaggle/hf_run_job.sh
#   HF_SPACE=you/arm-gym ./kaggle/hf_run_job.sh
#   HF_JOB_DETACH=0 ./kaggle/hf_run_job.sh   # stream logs in foreground (no -d)

set -euo pipefail

# 7B bf16+LoRA+GRPO fits a10g-large (24G). 14B+ use a100-large or set HF_FLAVOR.
: "${HF_FLAVOR:=a10g-large}"
: "${HF_JOB_TIMEOUT:=3h}"
: "${HF_SPACE:=kaori02/arm-gym}"
: "${HF_JOB_IMAGE:=huggingface/transformers-pytorch-gpu}"
# HF_MODEL is read by kaggle/hf_train_job.sh (not duplicated in HF_TRAIN_EXTRAS).
: "${HF_MODEL:=Qwen/Qwen2.5-Coder-7B-Instruct}"
# Match wiki SuperCoder: temp 0.5, lr 1e-6 (overrides train defaults when passed last).
: "${HF_TRAIN_EXTRAS:=--stack single_gpu --temperature 0.5 --num-generations 4 --learning-rate 1e-6}"
: "${HF_JOB_DETACH:=1}"

if ! command -v hf >/dev/null 2>&1; then
  echo "[hf_run_job] install:  pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

# $HF_TOKEN injected by --secrets HF_TOKEN. Single-quoted → expands on the VM.
# Hot-patches the cloned Space so the job works even if Space git is out of date.
# shellcheck disable=SC2016
read -r -d '' REMOTE <<'JOBSCRIPT' || true
set -euo pipefail
: "${HF_TOKEN:?log in: hf auth login, then re-run}"
# Job VM reads HF_MODEL for kaggle/hf_train_job.sh (set by local hf_run_job --env).
: "${HF_MODEL:=Qwen/Qwen2.5-Coder-7B-Instruct}"
export HF_MODEL
U="${HF_SPACE%%/*}"; R="${HF_SPACE#*/}"
git clone --depth 1 "https://user:${HF_TOKEN}@huggingface.co/spaces/${U}/${R}" /tmp/arm-gym-job
cd /tmp/arm-gym-job

echo "[hotpatch] fixing cloned Space files"
# 1. setup.sh: bare python/pip → python3
[ -f kaggle/setup.sh ] && {
  sed -i 's/^pip /python3 -m pip /' kaggle/setup.sh
  sed -i 's/^python - /python3 - /' kaggle/setup.sh
}

# 1b. train.py: (a) add top_p / top_k to generation_kwargs, (b) switch single_gpu
# loader from QLoRA-4bit to bf16. The 4-bit path is broken on the HF Jobs image
# because the bitsandbytes C++ extensions don't load with torch 2.10
# ("Skipping import of cpp extensions due to incompatible torch version"),
# and the Python fallback corrupts logits → multilingual gibberish.
[ -f kaggle/train.py ] && python3 - <<'PYPATCH'
import pathlib, re
p = pathlib.Path("kaggle/train.py")
src = p.read_text()

# (a) generation_kwargs: add top_p / top_k
needle_a = '"eos_token_id": list(QWEN25_CHAT_EOS_KNOWN_IDS),'
add_a = '"eos_token_id": list(QWEN25_CHAT_EOS_KNOWN_IDS),\n        "top_p": 0.9,\n        "top_k": 40,'
if needle_a in src and '"top_p":' not in src:
    src = src.replace(needle_a, add_a, 1)
    print("[hotpatch] train.py: added top_p=0.9 top_k=40 to generation_kwargs")
# Tighten older hotpatches (0.95/50) to wiki-aligned sampling
if '"top_p": 0.95' in src:
    src = src.replace('"top_p": 0.95', '"top_p": 0.9', 1)
    print("[hotpatch] train.py: top_p 0.95 -> 0.9")
if '"top_k": 50' in src:
    src = src.replace('"top_k": 50', '"top_k": 40', 1)
    print("[hotpatch] train.py: top_k 50 -> 40")
if "repetition_penalty=1.08" in src:
    src = src.replace("repetition_penalty=1.08", "repetition_penalty=1.1", 1)
    print("[hotpatch] train.py: repetition_penalty 1.08 -> 1.1")

# (b) single_gpu loader: replace QLoRA 4-bit with bf16 LoRA
old_block = (
    "    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training\n"
    "    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig\n"
    "\n"
    "    tok = AutoTokenizer.from_pretrained(cfg.model_id)\n"
    "    if tok.pad_token is None:\n"
    "        tok.pad_token = tok.eos_token\n"
    "\n"
    "    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=\"bfloat16\")\n"
    "    model = AutoModelForCausalLM.from_pretrained(\n"
    "        cfg.model_id, quantization_config=bnb, device_map=\"auto\",\n"
    "    )\n"
)
new_block = (
    "    import torch\n"
    "    from peft import LoraConfig, get_peft_model\n"
    "    from transformers import AutoModelForCausalLM, AutoTokenizer\n"
    "\n"
    "    tok = AutoTokenizer.from_pretrained(cfg.model_id)\n"
    "    if tok.pad_token is None:\n"
    "        tok.pad_token = tok.eos_token\n"
    "\n"
    "    model = AutoModelForCausalLM.from_pretrained(\n"
    "        cfg.model_id, torch_dtype=torch.bfloat16, device_map=\"auto\",\n"
    "    )\n"
)
if old_block in src:
    src = src.replace(old_block, new_block, 1)
    print("[hotpatch] train.py: switched single_gpu loader from 4-bit QLoRA to bf16 LoRA")
elif "torch_dtype=torch.bfloat16" in src:
    print("[hotpatch] train.py: bf16 loader already present")
else:
    print("[hotpatch] train.py: bf16 loader pattern not found; leaving 4-bit path")

# Replace prepare_model_for_kbit_training call with bf16-compatible setup
old_prep = (
    "    model = prepare_model_for_kbit_training(\n"
    "        model,\n"
    "        use_gradient_checkpointing=True,\n"
    "        gradient_checkpointing_kwargs={\"use_reentrant\": False},\n"
    "    )\n"
)
new_prep = (
    "    if hasattr(model, \"enable_input_require_grads\"):\n"
    "        model.enable_input_require_grads()\n"
    "    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={\"use_reentrant\": False})\n"
)
if old_prep in src:
    src = src.replace(old_prep, new_prep, 1)
    print("[hotpatch] train.py: replaced prepare_model_for_kbit_training with bf16 prep")

# Update the print label so we can verify in logs
src = src.replace('"[model] 4-bit loaded  ', '"[model] bf16 loaded  ')

# (c) Avoid GRPO cutting the *end* of the chat (loses "Optimized Assembly Code:")
# Default tokenizer trunc_side=right drops the generation anchor → gibberish.
if 'tok.truncation_side = "left"' not in src and "tok.pad_token = tok.eos_token" in src:
    src = src.replace(
        "    if tok.pad_token is None:\n        tok.pad_token = tok.eos_token\n",
        "    if tok.pad_token is None:\n        tok.pad_token = tok.eos_token\n    "
        "tok.truncation_side = \"left\"  # keep tail of prompt for GRPO\n",
        1,
    )
    print("[hotpatch] train.py: tokenizer truncation_side=left (keep prompt anchor)")
# (d) Raise default max_prompt_length so long baselines fit
src = src.replace("max_prompt_length: int = 1536", "max_prompt_length: int = 4096", 1)
if "max_prompt_length: int = 4096" in src:
    print("[hotpatch] train.py: max_prompt_length 1536 -> 4096 (if present)")
# (e) SuperCoder A.1: max response 2000 tokens
if "max_completion_length: int = 1024" in src:
    src = src.replace("max_completion_length: int = 1024", "max_completion_length: int = 2000", 1)
    print("[hotpatch] train.py: max_completion_length 1024 -> 2000 (paper)")

p.write_text(src)
PYPATCH

# 2. stopping criteria: replace with NO-OP (let model use natural chat EOS)
# The previous floor=32/64 was forcing the model to fill garbage padding,
# generating gibberish that GRPO could not learn from. Natural EOS works better.
cat > kaggle/assembly_stopping_criteria.py << 'STOPPY'
from __future__ import annotations
import torch
from transformers import StoppingCriteria

class StopOnAssemblyTag(StoppingCriteria):
    """No-op stub. We rely on chat EOS (im_end) instead of a tag-based stop."""
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        b = input_ids.shape[0]
        return torch.zeros(b, dtype=torch.bool, device=input_ids.device)
STOPPY

cat > kaggle/apply_model_generate_stopping_shim.py << 'SHIMPY'
"""No-op stopping shim. Let the model terminate naturally on chat EOS."""
from __future__ import annotations
from typing import Any

def install_grpo_assembly_stopping(model: Any, tokenizer: Any, **kwargs) -> None:
    print("[stopping-shim] DISABLED -- using natural chat EOS only", flush=True)
    return
SHIMPY
echo "[hotpatch] stopping criteria DISABLED -- using natural chat EOS"

# 3-4. kaggle/dataset.py + kaggle/reward_fn.py: from clone (push before job; no heredoc)
echo "[hotpatch] using kaggle/dataset.py + kaggle/reward_fn.py from clone"

# Run toolchain setup
[ "${HF_RUN_SETUP:-1}" = "1" ] && [ -f kaggle/setup.sh ] && bash kaggle/setup.sh

# Remove vllm/unsloth (incompatible with trl 0.20 import on single_gpu)
python3 -m pip uninstall -y vllm unsloth unsloth_zoo >/dev/null 2>&1 || true

export HF_SKIP_CLONE=1 HF_CLONE_DIR="${PWD}"
exec bash kaggle/hf_train_job.sh
JOBSCRIPT

cmdline=(hf jobs run
  --flavor "$HF_FLAVOR"
  --timeout "$HF_JOB_TIMEOUT"
  --env "HF_SPACE=$HF_SPACE"
  --env "HF_MODEL=$HF_MODEL"
  --env "HF_TRAIN_EXTRAS=$HF_TRAIN_EXTRAS"
  --secrets HF_TOKEN
)
if [ "$HF_JOB_DETACH" = 1 ]; then
  cmdline+=(-d)
fi
cmdline+=("$HF_JOB_IMAGE" bash -lc "$REMOTE")

echo "[hf_run_job] image=$HF_JOB_IMAGE  flavor=$HF_FLAVOR  HF_MODEL=$HF_MODEL  timeout=$HF_JOB_TIMEOUT  HF_SPACE=$HF_SPACE" >&2
exec "${cmdline[@]}"
