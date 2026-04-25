"""Evaluate the trained LoRA adapter against gcc -O3 baseline.

Usage:
  python scripts/evaluate.py \
      --checkpoint colab/results/runs/grpo/checkpoint-200 \
      --samples 8

Outputs:
  - Side-by-side assembly for each kernel
  - MCA cycles: baseline vs trained
  - Win rate (how often the model beats gcc -O3)
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

os.environ["PATH"] = "/opt/homebrew/opt/llvm/bin:" + os.environ.get("PATH", "")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from arm_gym.compile_baseline import detect_toolchain, compile_to_asm
from arm_gym.mca import run_mca
from arm_gym.kernels import generate_all, split_train_eval, TEMPLATES
from kaggle.dataset import SYSTEM_PROMPT, user_prompt


def parse_assembly(text: str) -> str | None:
    m = re.search(r"<assembly>(.*?)</assembly>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: look for .text section
    if ".text" in text or ".global" in text:
        return text.strip()
    return None


def load_model(checkpoint: Path, device: str):
    cfg_path = checkpoint / "adapter_config.json"
    import json
    base_model_id = json.loads(cfg_path.read_text())["base_model_name_or_path"]
    print(f"Base model: {base_model_id}")
    print(f"Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(checkpoint)

    print(f"Loading base model (float16)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    print(f"Loading LoRA adapter from {checkpoint}...")
    model = PeftModel.from_pretrained(model, str(checkpoint))
    model.eval()
    print("Model ready.\n")
    return model, tok


def generate(model, tok, c_source: str, baseline_asm: str, device: str, max_new_tokens: int = 512) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(c_source, baseline_asm)},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.5,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    default=PROJECT / "colab/results/runs/grpo/checkpoint-200")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--difficulty", type=int, default=1)
    args = ap.parse_args()

    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.checkpoint}\n")

    tc = detect_toolchain()
    print(f"Toolchain: clang={tc.clang} mca={tc.mca} mcpu={tc.mcpu}\n")
    assert tc.ready() and tc.mca, "Run: brew install llvm"

    model, tok = load_model(args.checkpoint, args.device)

    variants = [v for v in generate_all()
                if TEMPLATES[v.template_name].difficulty <= args.difficulty]
    _, eval_variants = split_train_eval(variants, eval_frac=0.1, seed=0)
    eval_variants = eval_variants[:args.samples]

    results = []
    sep = "-" * 72

    for i, v in enumerate(eval_variants):
        print(f"\n{'='*72}")
        print(f"[{i+1}/{len(eval_variants)}] {v.variant_id}")
        print(f"{'='*72}")

        # Baseline
        try:
            baseline_asm = compile_to_asm(v.c_source, tc)
            base_rep = run_mca(baseline_asm, tc.mca, tc.mcpu)
            base_cycles = base_rep.total_cycles
        except Exception as e:
            print(f"Baseline compile failed: {e}")
            continue

        # C source
        print("\nC Source:")
        print(v.c_source.strip())

        # Generate
        print(f"\nGenerating assembly (device={args.device})...")
        raw = generate(model, tok, v.c_source, baseline_asm, args.device)
        trained_asm = parse_assembly(raw)

        if trained_asm is None:
            print("Model output (no <assembly> tags found):")
            print(raw[:500])
            results.append({"variant": v.variant_id, "base": base_cycles,
                             "trained": None, "speedup": None, "status": "parse_fail"})
            continue

        # MCA on trained assembly
        try:
            tr_rep = run_mca(trained_asm, tc.mca, tc.mcpu)
            trained_cycles = tr_rep.total_cycles
            speedup = base_cycles / trained_cycles if trained_cycles > 0 else 0
            beat = speedup > 1.0
            status = "WIN" if beat else "LOSS"
        except Exception as e:
            print(f"MCA on trained asm failed: {e}")
            print("Trained asm:")
            print(trained_asm[:300])
            results.append({"variant": v.variant_id, "base": base_cycles,
                             "trained": None, "speedup": None, "status": "mca_fail"})
            continue

        print(f"\nBaseline assembly (gcc -O3, {base_cycles} cycles):")
        print(sep)
        for line in baseline_asm.strip().split("\n")[:20]:
            print(line)
        if baseline_asm.count("\n") > 20:
            print("  ... (truncated)")

        print(f"\nTrained assembly ({trained_cycles} cycles):")
        print(sep)
        for line in trained_asm.strip().split("\n")[:20]:
            print(line)
        if trained_asm.count("\n") > 20:
            print("  ... (truncated)")

        marker = "✓ BEATS gcc -O3" if beat else "✗ slower than gcc -O3"
        print(f"\n{sep}")
        print(f"Result: {status}  |  baseline={base_cycles}  trained={trained_cycles}  speedup={speedup:.2f}x  {marker}")

        results.append({"variant": v.variant_id, "base": base_cycles,
                         "trained": trained_cycles, "speedup": speedup, "status": status})

    # Summary
    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    valid = [r for r in results if r["speedup"] is not None]
    wins = [r for r in valid if r["status"] == "WIN"]
    parse_fails = sum(1 for r in results if r["status"] == "parse_fail")
    mca_fails = sum(1 for r in results if r["status"] == "mca_fail")

    print(f"{'Kernel':<35} {'Base':>6} {'Trained':>8} {'Speedup':>8} {'Result':>6}")
    print("-" * 72)
    for r in results:
        sp = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
        tr = str(r['trained']) if r['trained'] else "N/A"
        print(f"{r['variant'][:35]:<35} {r['base']:>6} {tr:>8} {sp:>8} {r['status']:>6}")

    print(f"\nWin rate:   {len(wins)}/{len(valid)} kernels beat gcc -O3 ({100*len(wins)/len(valid):.0f}%)" if valid else "No valid results")
    if wins:
        best = max(wins, key=lambda r: r["speedup"])
        print(f"Best:       {best['speedup']:.2f}x speedup on {best['variant']}")
    if valid:
        avg = sum(r["speedup"] for r in valid) / len(valid)
        print(f"Avg speedup: {avg:.2f}x across all valid kernels")
    if parse_fails:
        print(f"Parse fails: {parse_fails} (model didn't use <assembly> tags)")
    if mca_fails:
        print(f"MCA fails:  {mca_fails} (invalid assembly syntax)")

    print(f"\nHow to improve:")
    print(f"  - More steps (200 is early; 1000+ shows consistent speedup)")
    print(f"  - Larger model (7B on Colab with bitsandbytes fixed)")
    print(f"  - Difficulty curriculum (unlock stage 2+ kernels after 40% win rate)")


if __name__ == "__main__":
    main()
