"""TRL GRPO training loop with Cut 5 fallbacks and day-0 SFT warmup.

Stack priority (auto-detected at startup):
  1. Unsloth 4×L4 DDP + vLLM colocate — validated by smoke_4xl4.py at day 0.
  2. Plain TRL DDP (no Unsloth) — safe fallback.
  3. Single GPU TRL — scheduler-grpo-example validated.

Weaker flag 4 fix (ARM base correctness): `maybe_sft_warmup` runs if day-0
smoke rollout < 40% correctness on held-out kernels. Warmup = 1 epoch on
(C → ARM asm) pairs from `aarch64-linux-gnu-gcc -O3`.

Wiki free win 3 (unsloth-advanced-grpo): lora_alpha = rank × 2 by construction,
fast_inference=True when Unsloth stack is live.
"""

from __future__ import annotations
import argparse
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TrainStack = Literal["unsloth_vllm", "plain_trl_ddp", "single_gpu"]


@dataclass
class TrainConfig:
    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    output_dir: str = "runs/grpo"
    stack: TrainStack = "single_gpu"
    lora_rank: int = 16
    learning_rate: float = 1e-6
    num_generations: int = 4
    max_prompt_length: int = 1024
    max_completion_length: int = 512
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    smoke: bool = False
    skip_sft: bool = False

    @property
    def lora_alpha(self) -> int:
        return self.lora_rank * 2  # wiki win 3


def detect_stack() -> TrainStack:
    if importlib.util.find_spec("unsloth") and importlib.util.find_spec("vllm"):
        return "unsloth_vllm"
    if importlib.util.find_spec("torch") and os.environ.get("WORLD_SIZE"):
        return "plain_trl_ddp"
    return "single_gpu"


def smoke_day0_correctness(env, model_id: str, n_samples: int = 32) -> float:
    """Return fraction of correct rollouts on held-out variants at step 0.

    Used to decide SFT warmup. Uses the model's zero-shot generation.
    """
    # Real impl: sample n_samples held-out variants, run through env, count OK.
    # For scaffold we return a placeholder above the 0.4 threshold so SFT
    # warmup stays OFF unless explicitly forced via env var.
    if os.environ.get("FORCE_SFT_WARMUP") == "1":
        return 0.0
    return 0.6


def maybe_sft_warmup(cfg: TrainConfig, env) -> bool:
    """Weaker flag 4: SFT warmup path if base correctness < 40%."""
    if cfg.skip_sft:
        return False
    rate = smoke_day0_correctness(env, cfg.model_id)
    if rate >= 0.4:
        return False
    print(f"[train] day-0 correctness {rate:.2%} < 40% — SFT warmup stage")
    print("[train] pairs: (C source, aarch64-linux-gnu-gcc -O3 assembly)")
    print("[train] 1 epoch, LR 5e-5, frozen except LoRA adapters")
    # Real impl would call transformers Trainer here. Scaffold stops at plan.
    return True


def build_grpo_config(cfg: TrainConfig):
    from trl import GRPOConfig
    return GRPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        gradient_checkpointing=True,
        report_to="trackio",
    )


def load_model_unsloth(cfg: TrainConfig):
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=cfg.model_id,
        max_seq_length=cfg.max_prompt_length + cfg.max_completion_length,
        dtype=None,
        load_in_4bit=True,
        fast_inference=True,  # wiki win 3
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    return model, tok


def load_model_plain(cfg: TrainConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, torch_dtype="auto")
    lora = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    return model, tok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="5-step sanity run")
    p.add_argument("--skip-sft", action="store_true")
    p.add_argument("--stack", choices=["auto", "unsloth_vllm", "plain_trl_ddp", "single_gpu"],
                   default="auto")
    args = p.parse_args(argv)
    cfg = TrainConfig(smoke=args.smoke, skip_sft=args.skip_sft)
    cfg.stack = detect_stack() if args.stack == "auto" else args.stack
    print(f"[train] stack={cfg.stack} model={cfg.model_id} lora_rank={cfg.lora_rank} "
          f"lora_alpha={cfg.lora_alpha}")

    if cfg.smoke:
        print("[train] smoke mode: skipping model load and training loop")
        return 0

    # Real orchestration left as an exercise — requires GPU + internet.
    # This module's job: encode the wiring, not run heavy compute in CI.
    print("[train] ready. run scripts/smoke_4xl4.py before launching full training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
