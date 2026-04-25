# New Hugging Face Job (GRPO training)

## One command (from a machine with `hf` logged in)

```bash
cd /path/to/arm-gym
./kaggle/hf_run_job.sh
```

Uses `hf jobs run` — `huggingface/transformers-pytorch-gpu`, **`a10g-large` (24 GB)** with **Qwen2.5-Coder-7B** (SuperCoder-style defaults: `temp=0.5`, `lr=1e-6`, wiki prompt with `Optimized Assembly Code:` anchor). Override: `HF_FLAVOR=a100-large HF_MODEL=Qwen/Qwen2.5-Coder-14B-Instruct` for 14B, or `HF_FLAVOR=h200` + 32B, `HF_JOB_DETACH=0` (stream logs).

---

Create a new run manually at [Hugging Face Jobs](https://huggingface.co/jobs) (or your org’s Jobs page) and paste the following.

## Hardware (example)

| Field        | Suggested value |
|-------------|-----------------|
| **GPU**     | **A10G large (24 GB)** for default 7B bf16+LoRA; A100 80G for 14B; H200 for 32B |
| **Image**   | `huggingface/transformers-pytorch-gpu` (or a CUDA image you already use with `apt` for `kaggle/setup.sh`) |
| **Timeout** | Long enough for 50 steps (~30–60+ minutes, depends on image) |

## Secrets / environment

| Name        | Value |
|------------|--------|
| `HF_TOKEN` | Read token with access to the Space (Settings → Access tokens) |

Optional:

| Name         | Value |
|-------------|--------|
| `HF_SPACE`  | `YOUR_USER/arm-gym` (default in `hf_job_entry.sh` is `kaori02/arm-gym`) |
| `HF_MODEL`  | default `Qwen/…-7B-Instruct`; set `Qwen/…-14B` or `Qwen/…-32B` to match `HF_FLAVOR` VRAM. |
| `HF_TRAIN_EXTRAS` | Extra flags for `kaggle/train.py`, e.g. `--steps 100` (space-separated) |
| `HF_RUN_SETUP`    | `1` (default) runs `kaggle/setup.sh`. Set `0` only if the image already has QEMU/gcc-aarch64/LLVM-MCA. |

## Command (copy-paste)

**Recommended (works for private Spaces):** clone, then the same **inlined** steps as `./kaggle/hf_run_job.sh` (so you do not need `kaggle/hf_job_entry.sh` in the Space):

```bash
set -euo pipefail
: "${HF_TOKEN:?}"
export HF_SPACE="${HF_SPACE:-kaori02/arm-gym}"
U="${HF_SPACE%%/*}"; R="${HF_SPACE#*/}"
git clone --depth 1 "https://user:${HF_TOKEN}@huggingface.co/spaces/${U}/${R}" ./work
cd ./work
[ "${HF_RUN_SETUP:-1}" = "1" ] && [ -f kaggle/setup.sh ] && bash kaggle/setup.sh
export HF_SKIP_CLONE=1 HF_CLONE_DIR="${PWD}"
[ -n "${HF_TRAIN_EXTRAS:-}" ] || export HF_TRAIN_EXTRAS="--assembly-min-new-tokens-before-close 64"
exec bash kaggle/hf_train_job.sh
```

If `kaggle/hf_job_entry.sh` **is** in your Space, `bash kaggle/hf_job_entry.sh` is equivalent.

**Public Space only:** you can pull the entry script from the Hub without git:

```bash
curl -fsSL "https://huggingface.co/spaces/kaori02/arm-gym/raw/main/kaggle/hf_job_entry.sh" -o /tmp/hf_job_entry.sh && bash /tmp/hf_job_entry.sh
```

(Replace `kaori02/arm-gym` with your `user/repo` in the URL if different.)

**Repo already in the job image** (baked-in checkout at `$PWD`):

```bash
bash kaggle/hf_job_entry.sh
```

## What the job runs

1. `kaggle/setup.sh` — QEMU user-static, aarch64 GCC, LLVM (MCA).
2. `kaggle/hf_train_job.sh` — pip pin (`trl==0.20.0`, `transformers` range), `PYTHONPATH`, then:
   - `kaggle/train.py` — 7B, 50 steps, LoRA 32, `max_train` 128, `runs/grpo-kaggle` output.
3. Stopping + rewards expect a **non-trivial** `</assembly>` block; default `HF_TRAIN_EXTRAS` sets `--assembly-min-new-tokens-before-close 64` (see `RunConfig` in `kaggle/train.py`).

## After the run

Upload `runs/grpo-kaggle` (e.g. LoRA adapter + `log.csv`) to the Space or a model repo, or add an `huggingface-cli upload` step to your entry script if needed.
