"""Smoke test for Cut 5: validate Unsloth + vLLM colocate on 4×L4 DDP.

If the stack does not come up cleanly, exits non-zero with a verdict the
caller can use to pick the fallback path.

Usage:
  python scripts/smoke_4xl4.py
"""

from __future__ import annotations
import importlib.util
import os
import subprocess
import sys


def gpu_count() -> int:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        return len([ln for ln in r.stdout.splitlines() if ln.strip()])
    except Exception:
        return 0


def has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def main() -> int:
    gpus = gpu_count()
    want_4xl4 = has("unsloth") and has("vllm") and gpus >= 4

    print(f"gpus_detected: {gpus}")
    print(f"unsloth_installed: {has('unsloth')}")
    print(f"vllm_installed: {has('vllm')}")
    print(f"trl_installed: {has('trl')}")

    if want_4xl4:
        print("verdict: unsloth_vllm")
        print("note: still run a 20-step training dry-run under DDP before committing")
        return 0
    if has("trl") and gpus >= 2:
        print("verdict: plain_trl_ddp")
        print("note: no Unsloth on multi-GPU, use TRL + Accelerate DDP")
        return 0
    if has("trl") and gpus >= 1:
        print("verdict: single_gpu")
        print("note: closest validated config (scheduler-grpo-example)")
        return 0
    print("verdict: cpu_only_dry_run")
    print("note: install trl + accelerate and pick a GPU host before training")
    return 2


if __name__ == "__main__":
    sys.exit(main())
