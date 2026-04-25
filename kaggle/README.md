# Kaggle training bundle

Short-run GRPO training on 4xL4. Produces the 4 PNG plots judging needs, plus `log.csv` artifact.

## Kaggle notebook setup

1. Create new notebook. Accelerator: **GPU T4 x2** for smoke, **GPU L4 x4** for full run.
2. Internet: ON.
3. Persistence: ON if you want to resume.
4. Add dataset: none (kernel corpus is generated at runtime).

## Cells to paste

### Cell 1 - clone + install

```python
import subprocess, os
# clone the repo (push to GitHub first if private)
subprocess.run(["git", "clone", "https://github.com/ka-ori/arm-gym.git", "/kaggle/working/arm-gym"], check=True)
os.chdir("/kaggle/working/arm-gym")
subprocess.run(["bash", "kaggle/setup.sh"], check=True)
```

### Cell 2 - smoke (toolchain + dataset + reward loop)

```python
import subprocess
r = subprocess.run(["python", "kaggle/train.py", "--smoke"], capture_output=True, text=True)
print(r.stdout); print(r.stderr)
```

Must print `reward for baseline-parity completion: [0.0]`.

### Cell 3 - stack detection

```python
r = subprocess.run(["python", "scripts/smoke_4xl4.py"], capture_output=True, text=True)
print(r.stdout)
# read verdict line to pick --stack flag below
```

### Cell 3a - single-GPU short curve (T4 or L4 single)

```python
r = subprocess.run([
    "python", "kaggle/train.py",
    "--stack", "single_gpu",
    "--steps", "200",
    "--max-train", "128",
    "--num-generations", "4",
    "--difficulty-max", "1",
    "--out", "runs/short",
], capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-1000:])
```

Expect 45-90 min on T4. Produces log.csv + checkpoints.

### Cell 3b - 4xL4 DDP run (preferred)

```python
# use plain_trl_ddp - Unsloth multi-GPU is fragile, skip it
import subprocess
r = subprocess.run([
    "accelerate", "launch", "--multi_gpu", "--num_processes", "4",
    "kaggle/train.py",
    "--stack", "plain_trl_ddp",
    "--steps", "400",
    "--max-train", "256",
    "--num-generations", "4",
    "--out", "runs/4xl4",
], capture_output=True, text=True)
print(r.stdout[-3000:]); print(r.stderr[-1000:])
```

### Cell 4 - plots + display

```python
import subprocess, glob
from IPython.display import Image, display

subprocess.run(["python", "kaggle/plot_curves.py",
                "--log", "runs/short/log.csv",
                "--out", "artifacts/plots"], check=False)
for png in sorted(glob.glob("artifacts/plots/*.png")):
    print(png)
    display(Image(filename=png))
```

### Cell 5 - download artifacts

```python
import subprocess
from IPython.display import FileLink
subprocess.run(["zip", "-r", "runs.zip", "runs/", "artifacts/"], check=False)
FileLink("runs.zip")
```

## What to expect

- **Smoke**: must print `reward for baseline-parity completion: [0.0]`.
- **200-step single GPU**: reward curve moves above 0 somewhere between step 60-150 for vec_add/dot variants.
- **400-step 4xL4**: correctness pass rate climbs above 0.4, mean reward above 0 on stage-2 kernels.

## If training looks dead

1. Check `runs/*/config.json` - confirm LR 1e-6, num_generations >= 4.
2. If reward is always 0, assembler gate is rejecting everything.
   Run: `python -c "from kaggle.reward_fn import syntax_reward; print(syntax_reward(completions=['<assembly>mov x0, x0\nret\n</assembly>'], variant_id=['test'], baseline_asm=['']))"`
3. Baseline_asm missing from prompt? - correctness collapses.
4. Unsloth + DDP hang? - switch to `--stack plain_trl_ddp`.
5. GRPOConfig crash? - param-stripping fallback in grpo_config() handles unknown kwargs automatically.

## Files

- `setup.sh` - apt + pip bootstrap.
- `dataset.py` - kernel variant to prompt.
- `reward_fn.py` - 3 reward fns (syntax, correctness, speedup) with shared verify cache.
- `train.py` - TRL GRPO trainer with stack auto-select + CSV logger.
- `plot_curves.py` - 4 judging PNGs from log.csv.
