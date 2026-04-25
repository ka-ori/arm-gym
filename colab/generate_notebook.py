"""Generate local Jupyter notebook for ARM-Gym GRPO training on Apple Silicon (M3 Pro)."""
import json, os
from pathlib import Path

PROJ = str(Path(__file__).parent.parent)
OUT = os.path.join(PROJ, "colab", "arm_gym_grpo_colab.ipynb")

LLVM_BIN = "/opt/homebrew/opt/llvm/bin"


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
    "# ARM-Gym GRPO Training (Apple Silicon M3 Pro)\n"
    "\n"
    "Train Qwen2.5-Coder-7B to generate optimized AArch64 assembly that beats gcc -O3.\n"
    "\n"
    "**Hardware**: MacBook M3 Pro 18GB unified memory, MPS backend\n"
    "**Requirements**: Homebrew, Python 3.10+, ~15GB free RAM\n"
    "**Time**: ~60-120 min for 50 steps on M3 Pro"
))

# --- Cell 1: Homebrew LLVM ---
cells.append(md("## 1. Install ARM cross-compilation toolchain (Homebrew LLVM)"))
cells.append(code(f"""\
import subprocess, os, shutil

llvm_bin = "{LLVM_BIN}"

if not shutil.which("llvm-mca") and llvm_bin not in os.environ.get("PATH", ""):
    print("Installing LLVM via Homebrew (may take a few minutes)...")
    r = subprocess.run(["brew", "install", "llvm"], capture_output=True, text=True)
    if r.returncode != 0:
        print("brew install llvm failed:")
        print(r.stderr[-1000:])
    else:
        print(r.stdout[-500:])

if llvm_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = llvm_bin + ":" + os.environ["PATH"]

for tool in ["clang", "llvm-mca"]:
    path = shutil.which(tool)
    print(f"{{tool}}: {{path or 'NOT FOUND'}}")

print("toolchain done")
"""))

# --- Cell 2: Python deps ---
cells.append(md("## 2. Install training stack"))
cells.append(code("""\
import subprocess, sys
cmds = [
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"],
    [sys.executable, "-m", "pip", "install", "-q",
     "torch", "transformers", "trl>=0.16", "peft", "accelerate",
     "datasets", "pydantic", "numpy", "matplotlib", "httpx"],
]
for cmd in cmds:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"WARN: {' '.join(cmd[:5])}... failed")
        print(r.stderr[-500:])
    else:
        print(f"OK: {' '.join(cmd[:5])}...")
print("deps done")
"""))

# --- Cell 3: Path setup ---
cells.append(md("## 3. Set up arm_gym source path"))
cells.append(code("""\
import sys, os
from pathlib import Path

here = Path(os.getcwd())
for root in [here.parent, here, here.parent.parent]:
    if (root / "arm_gym" / "__init__.py").exists():
        sys.path.insert(0, str(root))
        print(f"arm_gym source at: {root}")
        break
else:
    raise RuntimeError("arm_gym not found - run notebook from arm-gym/ or arm-gym/colab/ directory")
"""))

# --- Cell 4: Smoke test ---
cells.append(md("## 4. Smoke test - verify toolchain + reward loop"))
cells.append(code(f"""\
import os, sys

llvm_bin = "{LLVM_BIN}"
if llvm_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = llvm_bin + ":" + os.environ["PATH"]

from arm_gym.compile_baseline import detect_toolchain
tc = detect_toolchain()
print(f"clang={{tc.clang}} gcc={{tc.gcc_aarch64}} mca={{tc.mca}} mcpu={{tc.mcpu}}")
assert tc.ready(), "toolchain not ready - rerun cell 1"
assert tc.mca, "llvm-mca not found - rerun cell 1"

from arm_gym.kernels import summary, generate_variants
s = summary()
print(f"templates={{s['templates']}} variants={{s['variants']}}")

from arm_gym.compile_baseline import compile_to_asm
v = next(generate_variants("vec_add"))
asm = compile_to_asm(v.c_source, tc)
print(f"compiled vec_add variant, asm length={{len(asm)}}")

from arm_gym.mca import run_mca
rep = run_mca(asm, tc.mca, tc.mcpu)
print(f"MCA: cycles={{rep.total_cycles}} ipc={{rep.ipc:.2f}}")
print("smoke OK")
"""))

# --- Cell 5: Device detection ---
cells.append(md("## 5. Device detection (MPS)"))
cells.append(code("""\
import torch

if torch.backends.mps.is_available():
    device = "mps"
    print(f"MPS available - Apple Silicon GPU acceleration enabled")
else:
    device = "cpu"
    print("WARNING: MPS not available, falling back to CPU (training will be very slow)")

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {device}")
print(f"float16 support: {torch.tensor([1.0], dtype=torch.float16).device}")
"""))

# --- Cell 6: Build dataset ---
cells.append(md("## 6. Build training dataset"))
cells.append(code(f"""\
import os

llvm_bin = "{LLVM_BIN}"
if llvm_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = llvm_bin + ":" + os.environ["PATH"]

from arm_gym.compile_baseline import detect_toolchain
from kaggle.dataset import DatasetConfig, build as build_dataset

tc = detect_toolchain()

DIFFICULTY = 1
MAX_TRAIN = 128
MAX_EVAL = 16

cfg = DatasetConfig(max_train=MAX_TRAIN, max_eval=MAX_EVAL, difficulty_max=DIFFICULTY)
train_ds, eval_ds, lookup = build_dataset(tc, cfg, tokenizer=None)
print(f"train={{len(train_ds)}} eval={{len(eval_ds)}} lookup={{len(lookup)}}")
print(f"sample prompt length: {{len(train_ds[0]['prompt'])}} chars")
print(f"sample variant_id: {{train_ds[0]['variant_id']}}")
"""))

# --- Cell 7: Training ---
cells.append(md(
    "## 7. GRPO Training\n"
    "\n"
    "Plain PEFT LoRA on MPS (Apple Silicon M3 Pro).\n"
    "- lora_rank=8, lora_alpha=16\n"
    "- num_generations=2, temperature=0.5\n"
    "- 50 steps (~60-120 min on M3 Pro)\n"
    "\n"
    "> **Memory note**: 7B in float16 uses ~14GB. Ensure no other large apps are open.\n"
    "> For a faster smoke test, switch MODEL_ID to `Qwen/Qwen2.5-Coder-1.5B-Instruct`."
))
cells.append(code(f"""\
import os, sys, time, csv, re
import torch
from pathlib import Path

llvm_bin = "{LLVM_BIN}"
if llvm_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = llvm_bin + ":" + os.environ["PATH"]

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
LORA_RANK = 8
LORA_ALPHA = 16
STEPS = 50
NUM_GENERATIONS = 2
MAX_PROMPT_LEN = 512
MAX_COMPLETION_LEN = 256
LR = 1e-6
BATCH_SIZE = 1
GRAD_ACCUM = 4
OUT_DIR = "runs/m3-grpo"

os.makedirs(OUT_DIR, exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {{device}}")

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model (float16, ~14GB)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map={{"": device}},
    low_cpu_mem_usage=True,
)
lora = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=0,
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()
print("model ready")

from arm_gym.compile_baseline import detect_toolchain
from kaggle.dataset import DatasetConfig, build as build_dataset

tc = detect_toolchain()
ds_cfg = DatasetConfig(max_train=128, max_eval=16, difficulty_max=1)
train_ds, eval_ds, _ = build_dataset(tc, ds_cfg, tokenizer=tokenizer)
print(f"dataset: train={{len(train_ds)}} eval={{len(eval_ds)}}")

from kaggle.reward_fn import syntax_reward, correctness_reward, speedup_reward
from trl import GRPOConfig, GRPOTrainer

grpo_params = dict(
    output_dir=OUT_DIR,
    max_steps=STEPS,
    learning_rate=LR,
    gradient_accumulation_steps=GRAD_ACCUM,
    per_device_train_batch_size=BATCH_SIZE,
    num_generations=NUM_GENERATIONS,
    max_prompt_length=MAX_PROMPT_LEN,
    max_completion_length=MAX_COMPLETION_LEN,
    gradient_checkpointing=False,
    bf16=False,
    fp16=False,
    max_grad_norm=0.1,
    temperature=0.5,
    beta=0.0,
    epsilon=0.2,
    remove_unused_columns=False,
    logging_steps=1,
    save_steps=25,
    save_total_limit=2,
    report_to="none",
    no_cuda=True,
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
        print(f"TRL compat: dropping {{dropped!r}}")
        grpo_params.pop(dropped, None)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[syntax_reward, correctness_reward, speedup_reward],
    args=gcfg,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tokenizer,
)

print(f"GRPO training: {{STEPS}} steps, lr={{LR}}, gen={{NUM_GENERATIONS}}")
start = time.time()
trainer.train()
elapsed = time.time() - start
print(f"training done in {{elapsed/60:.1f}} min")

log_path = Path(OUT_DIR) / "log.csv"
history = getattr(trainer.state, "log_history", [])
if history:
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    print(f"saved log to {{log_path}} ({{len(history)}} rows)")

model.save_pretrained(f"{{OUT_DIR}}/lora-adapter")
tokenizer.save_pretrained(f"{{OUT_DIR}}/lora-adapter")
print(f"saved LoRA adapter to {{OUT_DIR}}/lora-adapter")
"""))

# --- Cell 8: Plots ---
cells.append(md("## 8. Generate training evidence plots"))
cells.append(code("""\
import os, sys, glob
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

from kaggle.plot_curves import (
    load_rows, plot_training_loss, plot_reward_curve,
    plot_correctness_rate, plot_before_after,
)

log_path = Path("runs/m3-grpo/log.csv")
out_path = Path("artifacts/plots")
out_path.mkdir(parents=True, exist_ok=True)

if log_path.exists():
    rows = load_rows(log_path)
    print(f"loaded {len(rows)} log rows")
    plot_training_loss(rows, out_path / "training_loss.png")
    plot_reward_curve(rows, out_path / "reward_curve.png")
    plot_correctness_rate(rows, out_path / "correctness_rate.png")
    print("generated 3 training curve plots")
else:
    print(f"WARNING: {log_path} not found - run training cell first")

plot_before_after(out_path / "before_after_kernel.png")

for png in sorted(glob.glob(str(out_path / "*.png"))):
    print(f"saved: {png}")
"""))

# --- Cell 9: Results ---
cells.append(md("## 9. View results"))
cells.append(code("""\
import os, glob
from pathlib import Path

out_dir = Path("runs/m3-grpo")
print(f"Run directory: {out_dir.resolve()}")
for f in sorted(out_dir.rglob("*")):
    if f.is_file():
        size = f.stat().st_size
        print(f"  {f.relative_to(out_dir)}  ({size/1024:.1f} KB)")

plots = sorted(glob.glob("artifacts/plots/*.png"))
print(f"\\nPlots ({len(plots)}):")
for p in plots:
    print(f"  {os.path.abspath(p)}")
"""))

# --- Cell 10: Push to HF ---
cells.append(md(
    "## 10. (Optional) Push LoRA adapter to HuggingFace Hub\n"
    "\n"
    "Uncomment and set your HF token to push the trained adapter."
))
cells.append(code("""\
# from huggingface_hub import login
# login(token="hf_YOUR_TOKEN")
# model.push_to_hub("your-username/arm-gym-grpo-lora")
# tokenizer.push_to_hub("your-username/arm-gym-grpo-lora")
# print("pushed to HF Hub")
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
