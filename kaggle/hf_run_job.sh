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

# 3. dataset.py: overwrite with fixed version (no <assembly> pre-injection)
cat > kaggle/dataset.py << 'DATASETPY'
"""Dataset builder: kernel variant -> GRPO prompt."""
from __future__ import annotations
from dataclasses import dataclass
from datasets import Dataset
from arm_gym.compile_baseline import ToolchainInfo, compile_to_asm
from arm_gym.kernels import KernelVariant, generate_all, split_train_eval

SYSTEM_PROMPT = (
    "You are an expert AArch64 (aarch64-linux-gnu-gcc) assembly writer. "
    "Obey the user block exactly. Output only what is asked in the required tags."
)

def user_prompt(c_source: str, baseline_asm: str) -> str:
    return (
        "Given the following C code and assembly code, your task is to generate "
        "highly optimized AArch64 assembly code.\n\n"
        f"C Code:\n{c_source}\n\n"
        f"Assembly Code:\n{baseline_asm}\n\n"
        "Only output the optimized assembly code. Do not include any other text. "
        "Do not write any comments in the assembly code. "
        "Wrap the assembly code in <assembly></assembly> tags.\n\n"
        "Optimized Assembly Code:\n"
    )

def render_prompt(tokenizer, c_source: str, baseline_asm: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(c_source, baseline_asm)},
    ]
    if tokenizer is not None:
        kwargs: dict = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        try:
            prompt = tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            prompt = tokenizer.apply_chat_template(messages, **kwargs)
    else:
        prompt = f"{SYSTEM_PROMPT}\n\n{messages[1]['content']}\n"
    return prompt.rstrip()

@dataclass
class DatasetConfig:
    max_train: int = 256
    max_eval: int = 32
    difficulty_max: int = 2

def build(tc: ToolchainInfo, cfg: DatasetConfig | None = None,
          tokenizer=None) -> tuple[Dataset, Dataset, dict[str, KernelVariant]]:
    cfg = cfg or DatasetConfig()
    from arm_gym.kernels import TEMPLATES
    variants = [v for v in generate_all()
                if TEMPLATES[v.template_name].difficulty <= cfg.difficulty_max]
    train_v, eval_v = split_train_eval(variants, eval_frac=0.1, seed=0)
    train_v = train_v[:cfg.max_train]
    eval_v = eval_v[:cfg.max_eval]
    def to_row(v: KernelVariant) -> dict | None:
        try:
            baseline_asm = compile_to_asm(v.c_source, tc)
        except Exception:
            return None
        prompt = render_prompt(tokenizer, v.c_source, baseline_asm)
        return {"prompt": prompt, "variant_id": v.variant_id,
                "baseline_asm": baseline_asm, "c_source": v.c_source}
    train_rows = [r for r in (to_row(v) for v in train_v) if r is not None]
    eval_rows = [r for r in (to_row(v) for v in eval_v) if r is not None]
    lookup = {v.variant_id: v for v in train_v + eval_v}
    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows), lookup
DATASETPY
echo "[hotpatch] dataset.py replaced (no <assembly> pre-injection)"

# 4. reward_fn.py: overwrite with version expecting both tags
cat > kaggle/reward_fn.py << 'REWARDPY'
"""Reward functions for GRPOTrainer - both-tag aware.

Model now emits both <assembly> and </assembly> in completions.
format_reward rewards: +0.1 open tag, +0.1 close tag, +0.1 body >= 20 chars.
"""
from __future__ import annotations
import hashlib, re, threading
from dataclasses import dataclass
from arm_gym.compile_baseline import detect_toolchain
from arm_gym.verifier import run_correctness_qemu
from arm_gym.mca import run_mca
from arm_gym.rollout_budget import TestCase
from arm_gym.verifier import VerifierConfig, assemble, cleanup_temp_dirs

_ASM_RE = re.compile(r"<assembly>(.*?)</assembly>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

def extract_assembly(text: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    m = _ASM_RE.search(text)
    if m:
        return _clean_assembly(m.group(1))
    if "</assembly>" in text.lower():
        body = re.split(r"</assembly>", text, flags=re.IGNORECASE)[0]
        if "<assembly>" in body.lower():
            body = re.split(r"<assembly>", body, flags=re.IGNORECASE)[-1]
        return _clean_assembly(body)
    if "<assembly>" in text.lower():
        return _clean_assembly(re.split(r"<assembly>", text, flags=re.IGNORECASE)[-1])
    return _clean_assembly(text)

def _clean_assembly(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:asm|assembly|aarch64)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

@dataclass
class _Entry:
    assembles: bool = False
    runs: bool = False
    speedup: float = 0.0

_CACHE: dict[str, _Entry] = {}
_LOCK = threading.Lock()
_BASELINE: dict[str, float] = {}
_VCFG: VerifierConfig | None = None
_DEBUG_LIMIT = 5
_DEBUG_SHOWN = 0

def _cfg() -> VerifierConfig:
    global _VCFG
    if _VCFG is None:
        tc = detect_toolchain()
        _VCFG = VerifierConfig(
            mca_bin=tc.mca or "llvm-mca", assembler="aarch64-linux-gnu-as",
            linker="aarch64-linux-gnu-ld", qemu="qemu-aarch64-static", mcpu=tc.mcpu)
    return _VCFG

def _bcy(vid: str, basm: str) -> float:
    if vid not in _BASELINE:
        try:
            rep = run_mca(basm, _cfg().mca_bin, _cfg().mcpu)
            _BASELINE[vid] = float(rep.total_cycles)
        except Exception:
            _BASELINE[vid] = 1000.0
    return _BASELINE[vid]

def _key(text: str, vid: str) -> str:
    return hashlib.md5(f"{text}::{vid}".encode()).hexdigest()

def _run(text: str, vid: str, basm: str) -> _Entry:
    global _DEBUG_SHOWN
    k = _key(text, vid)
    with _LOCK:
        if k in _CACHE:
            return _CACHE[k]
    e = _Entry()
    asm = extract_assembly(text)
    cfg = _cfg()
    obj, err = assemble(asm, cfg)
    if err or obj is None:
        if _DEBUG_SHOWN < _DEBUG_LIMIT:
            with _LOCK:
                if _DEBUG_SHOWN < _DEBUG_LIMIT:
                    _DEBUG_SHOWN += 1
                    print(f"[reward-debug #{_DEBUG_SHOWN}] vid={vid} asm_fail={err.message[:200] if err else 'None'!r}", flush=True)
                    print(f"[reward-debug] raw[:300]={text[:300]!r}", flush=True)
                    print(f"[reward-debug] extracted[:300]={asm[:300]!r}", flush=True)
        cleanup_temp_dirs()
        with _LOCK:
            _CACHE[k] = e
        return e
    e.assembles = True
    e.runs = run_correctness_qemu(obj, TestCase(inputs=(), expected=None), cfg)
    if e.runs:
        bc = _bcy(vid, basm)
        try:
            rep = run_mca(asm, cfg.mca_bin, cfg.mcpu)
            e.speedup = bc / max(rep.total_cycles, 1)
        except Exception:
            e.speedup = 0.0
    cleanup_temp_dirs()
    with _LOCK:
        _CACHE[k] = e
    return e

def _prep(completions, kwargs):
    texts = [c[-1]["content"] if isinstance(c, list) else str(c) for c in (completions or [])]
    n = len(texts)
    vids = list(kwargs.get("variant_id") or [""] * n)
    basms = list(kwargs.get("baseline_asm") or [""] * n)
    if len(vids) == 1 and n > 1:
        vids = vids * n
        basms = basms * n
    return texts, vids, basms

def syntax_reward(prompts=None, completions=None, **kwargs) -> list[float]:
    _ = prompts
    texts, vids, basms = _prep(completions, kwargs)
    return [1.0 if _run(t, v, b).assembles else 0.0 for t, v, b in zip(texts, vids, basms)]

_FORMAT_DEBUG_STEP = 0
_FORMAT_DEBUG_LIMIT = 6

def format_reward(prompts=None, completions=None, **kwargs) -> list[float]:
    global _FORMAT_DEBUG_STEP
    _ = prompts
    texts, _, _ = _prep(completions, kwargs)
    if _FORMAT_DEBUG_STEP < _FORMAT_DEBUG_LIMIT and texts:
        _FORMAT_DEBUG_STEP += 1
        head = texts[0]
        print(f"[completion-debug step={_FORMAT_DEBUG_STEP}] len={len(head)} first 500 chars:", flush=True)
        print(repr(head[:500]), flush=True)
        print(f"[completion-debug step={_FORMAT_DEBUG_STEP}] last 200 chars:", flush=True)
        print(repr(head[-200:]), flush=True)
    scores = []
    for text in texts:
        lowered = text.lower()
        has_open = "<assembly>" in lowered
        has_close = "</assembly>" in lowered
        body = extract_assembly(text)
        body_len = len(re.sub(r"\s", "", body))
        has_prose = any(m in lowered for m in ("```", "<think>", "explain", "analysis"))
        score = 0.0
        if has_open:
            score += 0.1
        if has_close:
            score += 0.1
        if body_len >= 20:
            score += 0.1
        if has_prose:
            score -= 0.05
        scores.append(max(0.0, score))
    return scores

def correctness_reward(prompts=None, completions=None, **kwargs) -> list[float]:
    _ = prompts
    texts, vids, basms = _prep(completions, kwargs)
    return [1.0 if _run(t, v, b).runs else 0.0 for t, v, b in zip(texts, vids, basms)]

def speedup_reward(prompts=None, completions=None, **kwargs) -> list[float]:
    _ = prompts
    texts, vids, basms = _prep(completions, kwargs)
    scores = []
    for t, v, b in zip(texts, vids, basms):
        e = _run(t, v, b)
        scores.append(max(0.0, e.speedup - 1.0) if e.runs else 0.0)
    return scores

class LiveRewardFn:
    @classmethod
    def build(cls) -> "LiveRewardFn":
        return cls()
    def __call__(self, prompts=None, completions=None, **kwargs) -> list[float]:
        texts, vids, basms = _prep(completions, kwargs)
        out = []
        for t, v, b in zip(texts, vids, basms):
            e = _run(t, v, b)
            if not e.assembles: out.append(0.0)
            elif not e.runs: out.append(0.1)
            else: out.append(max(0.0, e.speedup - 1.0))
        return out
REWARDPY
echo "[hotpatch] reward_fn.py replaced (both-tag format_reward)"

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
