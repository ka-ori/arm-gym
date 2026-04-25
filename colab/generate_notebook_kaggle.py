"""Generate Kaggle notebook for ARM-Gym GRPO training (T4 GPU, 4-bit QLoRA)."""
import base64, io, json, os, tarfile
from pathlib import Path

PROJ = str(Path(__file__).parent.parent)
OUT = os.path.join(PROJ, "colab", "arm_gym_grpo_kaggle.ipynb")

# Build tarball of arm_gym + kaggle source
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for subdir in ["arm_gym", "kaggle"]:
        d = os.path.join(PROJ, subdir)
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.endswith(".py"):
                    full = os.path.join(root, f)
                    arcname = os.path.relpath(full, PROJ)
                    tar.add(full, arcname=arcname)
b64 = base64.b64encode(buf.getvalue()).decode()
print(f"tarball: {len(buf.getvalue())} bytes, b64: {len(b64)} chars")

chunk_size = 120
b64_lines = [b64[i:i+chunk_size] for i in range(0, len(b64), chunk_size)]
b64_literal = '(\n' + '\n'.join(f'    "{chunk}"' for chunk in b64_lines) + '\n)'


def md(lines):
    if isinstance(lines, str):
        lines = [l + "\n" for l in lines.strip().split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code(src):
    lines = [l + "\n" for l in src.strip().split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "metadata": {}, "source": lines,
            "outputs": [], "execution_count": None}


cells = []

cells.append(md(
    "# ARM-Gym GRPO Training (Kaggle T4)\n"
    "\n"
    "Train Qwen2.5-Coder-3B to generate optimized AArch64 assembly that beats gcc -O3.\n"
    "\n"
    "**Accelerator**: GPU T4 x1 or T4 x2 (single GPU used)\n"
    "**Internet**: ON (model download)\n"
    "**Time**: ~45-75 min for 200 steps"
))

# Cell 0: must run before any torch import
cells.append(md("## 0. Environment setup (run first)"))
cells.append(code("""\
import os

# Kaggle T4 x2 launches 2 distributed workers and sets these vars before the notebook runs.
# Unsetting them forces single-process mode in accelerate/transformers.
for _v in ["MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE",
           "TORCHELASTIC_RESTART_COUNT", "GROUP_RANK", "ROLE_RANK", "ROLE_NAME",
           "TORCHELASTIC_USE_AGENT_STORE"]:
    os.environ.pop(_v, None)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("single GPU mode: distributed vars cleared, GPU 0 only")
"""))

# --- Cell 1: Toolchain ---
cells.append(md("## 1. Install ARM cross-compilation toolchain"))
cells.append(code("""\
%%bash
set -e
echo "[1/2] apt toolchain"
apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  gcc-aarch64-linux-gnu binutils-aarch64-linux-gnu >/dev/null

echo "[2/2] llvm"
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq clang llvm llvm-dev >/dev/null
# Try llvm-21 from apt.llvm.org (best effort)
curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
  | gpg --dearmor -o /usr/share/keyrings/llvm.gpg 2>/dev/null || true
CODENAME=$(lsb_release -cs)
echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/${CODENAME}/ llvm-toolchain-${CODENAME}-21 main" \
  > /etc/apt/sources.list.d/llvm21.list 2>/dev/null || true
apt-get update -qq 2>/dev/null || true
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq clang-21 llvm-21 >/dev/null 2>&1 \
  && echo "llvm-21 installed" || echo "llvm-21 unavailable, using default"

echo "verify"
which aarch64-linux-gnu-gcc
which llvm-mca-21 2>/dev/null || which llvm-mca
echo "toolchain done"
"""))

# --- Cell 2: Python deps ---
cells.append(md("## 2. Install training stack"))
cells.append(code("""\
import subprocess, sys
cmds = [
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"],
    [sys.executable, "-m", "pip", "install", "-q",
     "transformers", "trl>=0.16", "peft", "accelerate",
     "datasets", "pydantic", "numpy", "matplotlib", "httpx"],
    [sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"],  # kept for optional use
]
for cmd in cmds:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"WARN: {' '.join(cmd[:5])}... failed")
        print(r.stderr[-300:])
    else:
        print(f"OK: {' '.join(cmd[:5])}...")
print("deps done")
"""))

# --- Cell 3: Unpack source ---
cells.append(md("## 3. Unpack arm_gym source"))
cells.append(code(f"""\
import base64, io, os, tarfile

B64 = {b64_literal}

data = base64.b64decode(B64)
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
    tar.extractall(".")
print("extracted arm_gym + kaggle source")
for root, dirs, files in os.walk("arm_gym"):
    for f in files:
        if f.endswith(".py"):
            print(f"  {{os.path.join(root, f)}}")
"""))

# --- Cell 4: Smoke test ---
cells.append(md("## 4. Smoke test"))
cells.append(code("""\
import os, sys, shutil

# Add llvm bin dir to PATH
for d in ["/usr/lib/llvm-21/bin", "/usr/lib/llvm-20/bin", "/usr/lib/llvm-14/bin"]:
    if os.path.isdir(d):
        os.environ["PATH"] = d + ":" + os.environ["PATH"]
        print(f"LLVM path: {d}")
        break

sys.path.insert(0, os.getcwd())

from arm_gym.compile_baseline import detect_toolchain
tc = detect_toolchain()
print(f"clang={tc.clang} gcc={tc.gcc_aarch64} mca={tc.mca} mcpu={tc.mcpu}")
assert tc.ready(), "toolchain not ready - rerun cell 1"
assert tc.mca, "llvm-mca not found - rerun cell 1"

from arm_gym.kernels import summary, generate_variants
s = summary()
print(f"templates={s['templates']} variants={s['variants']}")

from arm_gym.compile_baseline import compile_to_asm
v = next(generate_variants("vec_add"))
asm = compile_to_asm(v.c_source, tc)
print(f"compiled vec_add, asm length={len(asm)}")

from arm_gym.mca import run_mca
rep = run_mca(asm, tc.mca, tc.mcpu)
print(f"MCA: cycles={rep.total_cycles} ipc={rep.ipc:.2f}")
print("smoke OK")
"""))

# --- Cell 5: GPU check ---
cells.append(md("## 5. GPU detection"))
cells.append(code("""\
import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory/1e9:.1f} GB")
else:
    print("WARNING: no GPU - enable T4 in notebook settings")
"""))

# --- Cell 6: Build dataset ---
cells.append(md("## 6. Build training dataset"))
cells.append(code("""\
import os, sys

for d in ["/usr/lib/llvm-21/bin", "/usr/lib/llvm-20/bin", "/usr/lib/llvm-14/bin"]:
    if os.path.isdir(d):
        os.environ["PATH"] = d + ":" + os.environ["PATH"]
        break

sys.path.insert(0, os.getcwd())

from arm_gym.compile_baseline import detect_toolchain
from kaggle.dataset import DatasetConfig, build as build_dataset

tc = detect_toolchain()

cfg = DatasetConfig(max_train=128, max_eval=16, difficulty_max=1)
train_ds, eval_ds, lookup = build_dataset(tc, cfg, tokenizer=None)
print(f"train={len(train_ds)} eval={len(eval_ds)} lookup={len(lookup)}")
print(f"sample prompt length: {len(train_ds[0]['prompt'])} chars")
"""))

# --- Cell 7: Training ---
cells.append(md(
    "## 7. GRPO Training\n"
    "\n"
    "PEFT LoRA, float16, single T4.\n"
    "- Model: Qwen2.5-Coder-3B-Instruct (~6GB VRAM)\n"
    "- lora_rank=16, lora_alpha=32, all 7 attention+MLP modules\n"
    "- num_generations=8, temperature=0.8\n"
    "- 500 steps (~100-150 min on T4)"
))
cells.append(code("""\
import os, sys, time, csv, re
import torch
from pathlib import Path

for d in ["/usr/lib/llvm-21/bin", "/usr/lib/llvm-20/bin", "/usr/lib/llvm-14/bin"]:
    if os.path.isdir(d):
        os.environ["PATH"] = d + ":" + os.environ["PATH"]
        break

sys.path.insert(0, os.getcwd())

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
LORA_RANK = 16
LORA_ALPHA = 32
STEPS = 500
NUM_GENERATIONS = 8
MAX_PROMPT_LEN = 1024
MAX_COMPLETION_LEN = 512
LR = 1e-6
BATCH_SIZE = 1
GRAD_ACCUM = 4
OUT_DIR = "/kaggle/working/runs/grpo"

os.makedirs(OUT_DIR, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

import gc
gc.collect()
torch.cuda.empty_cache()
free, total = torch.cuda.mem_get_info(0)
print(f"GPU 0 memory: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
if free < 8e9:
    raise RuntimeError(
        f"Only {free/1e9:.1f} GB free on GPU 0. "
        "Restart the Kaggle kernel (Run > Restart Session) then run all cells from Cell 0."
    )

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model (float16, ~6GB VRAM)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)
lora = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()
print("model ready")

from arm_gym.compile_baseline import detect_toolchain
from kaggle.dataset import DatasetConfig, build as build_dataset

tc = detect_toolchain()
ds_cfg = DatasetConfig(max_train=128, max_eval=16, difficulty_max=1)
train_ds, eval_ds, _ = build_dataset(tc, ds_cfg, tokenizer=tokenizer)
print(f"dataset: train={len(train_ds)} eval={len(eval_ds)}")

from kaggle.reward_fn import syntax_reward, correctness_reward, speedup_reward
from trl import GRPOConfig, GRPOTrainer

grpo_params = dict(
    output_dir=OUT_DIR,
    max_steps=STEPS,
    learning_rate=LR,
    gradient_accumulation_steps=GRAD_ACCUM,
    per_device_train_batch_size=BATCH_SIZE,
    num_generations=NUM_GENERATIONS,
    generation_batch_size=NUM_GENERATIONS,
    max_completion_length=MAX_COMPLETION_LEN,
    gradient_checkpointing=True,
    bf16=False,
    fp16=False,
    max_grad_norm=0.1,
    temperature=0.8,
    beta=0.0,
    epsilon=0.2,
    remove_unused_columns=False,
    logging_steps=1,
    save_steps=100,
    save_total_limit=2,
    report_to="none",
)

while True:
    try:
        gcfg = GRPOConfig(**grpo_params)
        break
    except TypeError as exc:
        m = re.search(r"unexpected keyword argument '(\\w+)'", str(exc))
        if not m:
            raise
        dropped = m.group(1)
        print(f"TRL compat: dropping {dropped!r}")
        grpo_params.pop(dropped, None)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[syntax_reward, correctness_reward, speedup_reward],
    args=gcfg,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)

print(f"GRPO training: {STEPS} steps, lr={LR}, gen={NUM_GENERATIONS}")
start = time.time()
trainer.train()
elapsed = time.time() - start
print(f"training done in {elapsed/60:.1f} min")

log_path = Path(OUT_DIR) / "log.csv"
history = getattr(trainer.state, "log_history", [])
if history:
    all_keys = list(dict.fromkeys(k for row in history for k in row.keys()))
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(history)
    print(f"saved log to {log_path} ({len(history)} rows)")

model.save_pretrained(f"{OUT_DIR}/lora-adapter")
tokenizer.save_pretrained(f"{OUT_DIR}/lora-adapter")
print(f"saved LoRA adapter to {OUT_DIR}/lora-adapter")
"""))

# --- Cell 8: Plots ---
cells.append(md("## 8. Training evidence plots"))
cells.append(code("""\
import os, sys, glob
sys.path.insert(0, os.getcwd())
from pathlib import Path

from kaggle.plot_curves import (
    load_rows, plot_training_loss, plot_reward_curve,
    plot_correctness_rate, plot_before_after,
)
from IPython.display import Image, display

log_path = Path("/kaggle/working/runs/grpo/log.csv")
out_path = Path("/kaggle/working/artifacts/plots")
out_path.mkdir(parents=True, exist_ok=True)

if log_path.exists():
    rows = load_rows(log_path)
    print(f"loaded {len(rows)} log rows")
    plot_training_loss(rows, out_path / "training_loss.png")
    plot_reward_curve(rows, out_path / "reward_curve.png")
    plot_correctness_rate(rows, out_path / "correctness_rate.png")
    print("generated 3 training curve plots")
    for png in sorted(glob.glob(str(out_path / "*.png"))):
        display(Image(filename=png))
else:
    print(f"WARNING: {log_path} not found - run training cell first")

plot_before_after(out_path / "before_after_kernel.png")
"""))

# --- Cell 9: Save output ---
cells.append(md(
    "## 9. Output\n"
    "\n"
    "All outputs saved to `/kaggle/working/` and available in the Output tab."
))
cells.append(code("""\
import subprocess
subprocess.run(["zip", "-r", "/kaggle/working/arm_gym_results.zip",
                "/kaggle/working/runs/", "/kaggle/working/artifacts/"], check=False)
print("arm_gym_results.zip saved to /kaggle/working/")
print("Download from the Output tab on the right.")
"""))

# --- Cell 10: HF push ---
cells.append(md(
    "## 10. (Optional) Push LoRA adapter to HuggingFace Hub"
))
cells.append(code("""\
# from huggingface_hub import login
# login(token="hf_YOUR_TOKEN")
# model.push_to_hub("your-username/arm-gym-grpo-lora")
# tokenizer.push_to_hub("your-username/arm-gym-grpo-lora")
print("uncomment above to push adapter to HF Hub")
"""))

# --- Build notebook ---
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"wrote {OUT}")
