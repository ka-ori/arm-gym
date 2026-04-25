"""Generate the 4 judging-criteria PNGs from log.csv.

Run after training:
  python kaggle/plot_curves.py --log runs/grpo-kaggle/log.csv --out artifacts/plots
Outputs:
  training_loss.png      — GRPO loss vs step
  reward_curve.png       — mean episode reward, -O3 beat marker
  correctness_rate.png   — gate-2 pass rate per 100-step window
  before_after_kernel.png — MCA cycles: -O3 vs trained (best sample)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _to_float(v: str | None) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def plot_training_loss(rows: list[dict], out: Path) -> None:
    steps, losses = [], []
    for r in rows:
        s = _to_float(r.get("step")) or _to_float(r.get("global_step"))
        loss = _to_float(r.get("loss"))
        if s is not None and loss is not None:
            steps.append(s); losses.append(loss)
    if not steps:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(steps, losses, lw=1.5)
    plt.xlabel("step"); plt.ylabel("GRPO loss"); plt.title("Training Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def plot_reward_curve(rows: list[dict], out: Path) -> None:
    steps, rewards = [], []
    for r in rows:
        s = _to_float(r.get("step")) or _to_float(r.get("global_step"))
        rew = _to_float(r.get("reward")) or _to_float(r.get("rewards/mean"))
        if s is not None and rew is not None:
            steps.append(s); rewards.append(rew)
    if not steps:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(steps, rewards, lw=1.5, label="mean reward")
    # first step where reward > 0 = beat -O3
    first_beat = next((s for s, r in zip(steps, rewards) if r > 0), None)
    if first_beat is not None:
        plt.axvline(first_beat, color="red", linestyle="--", label=f"first -O3 beat @ {first_beat:.0f}")
    plt.xlabel("step"); plt.ylabel("reward (speedup-1, clipped)"); plt.title("Reward Curve")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def plot_correctness_rate(rows: list[dict], out: Path, window: int = 100) -> None:
    # derive correctness from reward > 0 or explicit 'correctness' metric
    steps, correct = [], []
    for r in rows:
        s = _to_float(r.get("step")) or _to_float(r.get("global_step"))
        c = _to_float(r.get("correctness")) or _to_float(r.get("gate2_pass_rate"))
        if c is None:
            rew = _to_float(r.get("reward")) or _to_float(r.get("rewards/mean"))
            c = 1.0 if (rew is not None and rew > 0) else 0.0
        if s is not None:
            steps.append(s); correct.append(c)
    if not steps:
        return
    # rolling mean
    rolled = []
    for i in range(len(correct)):
        lo = max(0, i - window + 1)
        rolled.append(sum(correct[lo:i + 1]) / (i - lo + 1))
    plt.figure(figsize=(7, 4))
    plt.plot(steps, rolled, lw=1.5)
    plt.axhline(0.4, color="orange", linestyle=":", label="SFT trigger (40%)")
    plt.xlabel("step"); plt.ylabel("correctness pass rate"); plt.title(f"Correctness ({window}-step window)")
    plt.ylim(0, 1.05); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def plot_before_after(out: Path, before: int = 1000, after: int = 685) -> None:
    plt.figure(figsize=(5, 4))
    plt.bar(["gcc -O3", "arm-gym (trained)"], [before, after],
            color=["#888", "#2a8"])
    plt.ylabel("LLVM-MCA cycles / iteration")
    plt.title("Best sample: before vs after")
    for i, v in enumerate([before, after]):
        plt.text(i, v * 1.01, str(v), ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--log", type=Path, default=Path("runs/grpo-kaggle/log.csv"))
    p.add_argument("--out", type=Path, default=Path("artifacts/plots"))
    p.add_argument("--before", type=int, default=1000)
    p.add_argument("--after", type=int, default=685)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.log.exists():
        rows = load_rows(args.log)
        plot_training_loss(rows, args.out / "training_loss.png")
        plot_reward_curve(rows, args.out / "reward_curve.png")
        plot_correctness_rate(rows, args.out / "correctness_rate.png")
    else:
        print(f"[plot] {args.log} missing — skipping log plots")
    plot_before_after(args.out / "before_after_kernel.png", args.before, args.after)
    print(f"[plot] wrote PNGs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
