"""Kaggle-ready GRPO trainer for arm-gym.

Run:
  !bash kaggle/setup.sh
  !python kaggle/train.py --smoke           # dry run, CPU ok
  !python kaggle/train.py --steps 200       # full run
  !accelerate launch kaggle/train.py --stack plain_trl_ddp --steps 200   # 2xT4

Auto stack selection:
  - Unsloth + vLLM if both importable (preferred)
  - Plain TRL DDP under accelerate if WORLD_SIZE > 1
  - Single-GPU fallback
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from arm_gym.compile_baseline import detect_toolchain

try:
    from .dataset import DatasetConfig, build as build_dataset
    from .reward_fn import (
        LiveRewardFn,
        correctness_reward,
        format_reward,
        speedup_reward,
        syntax_reward,
    )
    from . import reward_fn as _reward_fn_mod
except ImportError:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from dataset import DatasetConfig, build as build_dataset
    from reward_fn import LiveRewardFn, correctness_reward, format_reward, speedup_reward, syntax_reward
    import reward_fn as _reward_fn_mod

# Defensive access to the reward cache — only used for memory trimming.
def _get_reward_cache() -> dict:
    return getattr(_reward_fn_mod, "_CACHE", {})


@dataclass
class RunConfig:
    # Parallel-experiment branch. Same base model as our teammate's Kaggle run
    # so the comparison is apples-to-apples on capacity, but deliberately
    # divergent on (loss, LoRA shape, group size, temperature) and reward stack
    # so the team covers two different optimization regimes simultaneously.
    # Default: Qwen2.5-Coder-7B (SuperCoder / hackathon wiki). 14B+ needs A100
    # headroom; override --model and GPU when scaling up.
    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    stack: str = "auto"                # auto | unsloth_vllm | plain_trl_ddp | single_gpu
    lora_rank: int = 32                # divergent: 32 vs teammate's 48
    # During GRPO rollout TRL keeps the model in train() mode (it needs grads on
    # the policy forward). With dropout > 0 the LoRA branches sample a different
    # mask each forward → first-step rollouts drift into rare-token loops even
    # when sanity-greedy and sanity-sampled both decode cleanly. Set to 0 for
    # deterministic rollouts; LoRA at rank=32 already has enough regularisation
    # from the small adapter size + bf16 noise.
    lora_dropout: float = 0.0
    # Only the unsloth_vllm stack still uses 4-bit; single_gpu is bf16 (no bnb).
    load_in_4bit: bool = False
    # SuperCoder A.2 / wiki: 1e-6 — assembly is token-precise; higher LR
    # destabilises valid syntax (gibberish, broken mnemonics).
    learning_rate: float = 1e-6
    # divergent: 4 vs teammate's 8. 16 % 4 == 0 (must divide
    # per_device_batch * grad_accum * world). 4 generations per prompt → smaller
    # group variance, but 4 prompts per gradient step instead of teammate's 2
    # → more prompt diversity per update.
    num_generations: int = 2
    # Wiki SuperCoder: 0.5 for GRPO (diversity without long-tail noise).
    temperature: float = 0.5
    warmup_steps: int = 20
    lr_scheduler_type: str = "constant_with_warmup"
    # Long C + long gcc -O3 baseline + wiki anchor; 1536 tok often truncates the
    # *end* of the user message (default trunc_side=right) → model never sees
    # "Optimized Assembly Code:" → degenerate gibberish. Use 2048 + left trunc.
    # 4096 makes attention cost quadratic; 2048 is enough for the longest
    # baseline asm in the dataset (gemv/matmul ≤ ~1500 tokens).
    max_prompt_length: int = 2048
    # 768 covers ≥99% of optimized rollouts (typical Coder asm is 200-500 tok)
    # while keeping logits memory bounded:
    #   logits[bs=2, T=768, V=152k, bf16] ≈ 0.46 GB / chunk.
    max_completion_length: int = 768
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    steps: int = 50
    save_steps: int = 25
    log_every: int = 1
    out_dir: str = "runs/grpo-kaggle"
    smoke: bool = False
    difficulty_max: int = 2
    max_train: int = 256
    max_eval: int = 32
    # Stopping: require this many *new* tokens before ``</assembly>`` can end
    # generation. Too low → 4-token "close only" → correctness/speedup stay 0.
    assembly_min_new_tokens_before_close: int = 32

    @property
    def lora_alpha(self) -> int:
        # Divergent from teammate: 2:1 alpha:rank to give a smaller-rank LoRA
        # the same effective update magnitude as their rank-48 / alpha-48 run.
        return self.lora_rank * 2


# Qwen2.5: im_end=151645, <|endoftext|>=151643 (tokenizer added_tokens)
QWEN25_CHAT_EOS_KNOWN_IDS: tuple[int, int] = (151645, 151643)


def _is_qwen_model_id(model_id: str) -> bool:
    m = (model_id or "").lower()
    return "qwen" in m


def _apply_qwen25_chat_eos_for_grpo(model_id: str, tok: Any, model: Any) -> None:
    """Align tokenizer + model stops with Qwen2.5 chat (assistant im_end) so ``generate`` halts on chat EOS.

    TRL >=0.20 builds completion sampling from ``PreTrainedTokenizer.eos_token_id``
    (see ``GRPOTrainer``), *not* from ``model.generation_config``. Setting only the model
    is insufficient. We also set ``GRPOConfig.generation_kwargs`` so the merged
    ``transformers.GenerationConfig`` cannot miss multi-token EOS.
    """
    if not _is_qwen_model_id(model_id):
        return

    im_end = "<|" + "im" + "_end|>"
    eos_ids: list[int] = []
    for tok_str in (im_end, "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(tok_str)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            if tid not in eos_ids:
                eos_ids.append(tid)
    if not eos_ids:
        eos_ids = list(QWEN25_CHAT_EOS_KNOWN_IDS)

    # Tokenizer: only a *single* int is valid here — assigning a list hits
    # ``Cannot set a non-string value as the eos_token`` in
    # ``tokenization_utils_base`` (eos_token_id is tied to the string
    # ``eos_token``). TRL’s GRPO build reads this scalar; use the chat
    # assistant stop (|im_end|) as the canonical one.
    primary_eos = eos_ids[0]
    tok.eos_token_id = primary_eos

    # model.generate / GenerationConfig accept int | list[int] for multiple stops
    eid: int | list[int] = eos_ids[0] if len(eos_ids) == 1 else eos_ids
    model.config.eos_token_id = primary_eos
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        gc.eos_token_id = eid


def _qwen25_grpo_generation_eos_kwargs(cfg: RunConfig) -> dict[str, Any] | None:
    if not _is_qwen_model_id(cfg.model_id):
        return None
    # ONLY override what's needed: chat-EOS list. Do NOT pass top_p/top_k/
    # repetition_penalty here — TRL builds its GenerationConfig from
    # GRPOConfig.temperature plus this dict, and:
    #  - top_p/top_k often warn "not valid and may be ignored" depending on
    #    transformers version; we'd rather rely on temperature alone
    #  - repetition_penalty=1.1 catastrophically penalises asm vocabulary
    #    (mov/add/ret/x0/...) when the prompt contains the full gcc -O3
    #    baseline asm, pushing rollouts to rare 'ampie/odzi/darm' tail tokens.
    return {"eos_token_id": list(QWEN25_CHAT_EOS_KNOWN_IDS)}


# ── progress + CSV callback ──────────────────────────────────────────────────

class StepLogger:
    """Writes one CSV row per log event; flushes immediately so no data is lost on crash."""

    def __init__(self, path: Path, total_steps: int):
        self.path = path
        self.total = total_steps
        self.start = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w", newline="", buffering=1)
        self.writer: csv.DictWriter | None = None
        self.fieldnames: list[str] = []

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        elapsed = time.time() - self.start
        rate = step / elapsed if elapsed > 0 and step > 0 else 0
        eta = (self.total - step) / rate if rate > 0 else 0
        pct = step / self.total * 100

        reward   = _fmt(metrics.get("reward"))
        syntax   = _fmt(metrics.get("rewards/syntax_reward/mean"))
        correct  = _fmt(metrics.get("rewards/correctness_reward/mean"))
        speedup  = _fmt(metrics.get("rewards/speedup_reward/mean"))
        loss     = _fmt(metrics.get("loss"))

        print(
            f"[step {step:>3}/{self.total}] {pct:5.1f}% | "
            f"elapsed={elapsed/60:5.1f}m  eta={eta/60:5.1f}m | "
            f"loss={loss}  reward={reward}  "
            f"syntax={syntax}  correct={correct}  speedup={speedup}",
            flush=True,
        )

        row = {"step": step, "elapsed_s": f"{elapsed:.1f}", **metrics}
        new_keys = [k for k in row if k not in self.fieldnames]
        if new_keys:
            self.fieldnames.extend(new_keys)
            # Rebuild writer with updated fieldnames
            self.writer = csv.DictWriter(
                self.f, fieldnames=self.fieldnames, extrasaction="ignore"
            )
            if step == 0 or (step == 1 and not self.f.tell()):
                self.writer.writeheader()

        if self.writer:
            self.writer.writerow({k: row.get(k, "") for k in self.fieldnames})
            self.f.flush()

    def close(self) -> None:
        self.f.close()


def _fmt(v: Any) -> str:
    if v is None or v == "":
        return "  -  "
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def _vram_str() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"{used:.1f}/{total:.1f}GB"
    except Exception:
        pass
    return "n/a"


# ── TRL callback ──────────────────────────────────────────────────────────────

def make_callback(logger: StepLogger, total_steps: int, cache_flush_every: int = 20):
    from transformers import TrainerCallback

    class _CB(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                logger.log(state.global_step, logs)

        def on_step_end(self, args, state, control, **kwargs):
            step = state.global_step

            # periodic memory cleanup to prevent CUDA cache buildup
            if step % cache_flush_every == 0 and step > 0:
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # trim reward cache — it grows unbounded across all rollouts
                cache = _get_reward_cache()
                if len(cache) > 5000:
                    keys = list(cache.keys())
                    for k in keys[:-2000]:
                        cache.pop(k, None)

                print(
                    f"  [mem] step={step} vram={_vram_str()} "
                    f"reward_cache={len(_get_reward_cache())} gc done",
                    flush=True,
                )

        def on_train_end(self, args, state, control, **kwargs):
            print(f"\n[done] training finished at step {state.global_step}", flush=True)

    return _CB()


# ── stack / model loading ─────────────────────────────────────────────────────

def resolve_stack(cfg: RunConfig) -> str:
    if cfg.stack != "auto":
        return cfg.stack
    has_uns = importlib.util.find_spec("unsloth") is not None
    has_vllm = importlib.util.find_spec("vllm") is not None
    if has_uns and has_vllm:
        return "unsloth_vllm"
    if os.environ.get("WORLD_SIZE") and int(os.environ["WORLD_SIZE"]) > 1:
        return "plain_trl_ddp"
    return "single_gpu"


def _sanity_check_inference(model: Any, tok: Any, cfg: RunConfig) -> None:
    """Generate 2 completions (greedy + sampled) and assert no gibberish.

    The previous run passed greedy sanity but emitted gibberish under the
    training-time sampler. This now runs both:
      1. greedy on hand-crafted prompt (logit baseline)
      2. sampled (T=cfg.temperature, top_p=0.9) on a real dataset prompt
         (mirrors the actual training rollout sampling path)

    Fails fast with exit 3 on either gibberish signature.
    """
    import torch
    bad_markers = ("ampie", "darm", "Hindered", "ReactDOM", "eczy", "odzi", "/*\n//import")

    def _decode_and_check(prompt: str, label: str, *, sample: bool) -> None:
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        model.eval()
        with torch.no_grad():
            gen_kwargs = dict(
                max_new_tokens=128,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            if sample:
                gen_kwargs.update(
                    do_sample=True,
                    temperature=cfg.temperature,
                    top_p=0.9,
                    # no repetition_penalty — same rationale as GRPOConfig (asm repeats tokens)
                )
            else:
                gen_kwargs["do_sample"] = False
            out = model.generate(**inputs, **gen_kwargs)
        model.train()
        text = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
        print(f"[sanity:{label}] {text[:400]!r}", flush=True)
        hits = sum(text.count(m) for m in bad_markers)
        if hits >= 2:
            raise SystemExit(
                f"[sanity:{label}] FAIL — gibberish hits={hits}. "
                f"attn={getattr(model.config, 'attn_implementation', '?')} "
                f"dtype={model.dtype}. "
                f"Likely PEFT/transformers/bnb ABI corruption. Decoded: {text[:300]!r}"
            )

    greedy_prompt = (
        "<|im_start|>system\nYou are an AArch64 assembly writer. Output only "
        "<assembly>...</assembly>.<|im_end|>\n<|im_start|>user\n"
        "Emit AArch64 assembly that returns 0 from `int kernel(void)`.\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )
    _decode_and_check(greedy_prompt, "greedy", sample=False)

    # Sampled check using the FIRST real training prompt (full kernel C + full
    # gcc -O3 baseline asm). The previous run-time gibberish was specifically
    # triggered on this large-prompt path; a synthetic small prompt missed it.
    try:
        from arm_gym.compile_baseline import detect_toolchain
        from arm_gym.kernels import generate_all
        try:
            from kaggle.dataset import render_prompt
        except ImportError:
            from dataset import render_prompt  # type: ignore
        v = generate_all()[0]
        from arm_gym.compile_baseline import compile_to_asm
        sample_prompt = render_prompt(
            tok,
            c_source=v.c_source,
            baseline_asm=compile_to_asm(v.c_source, detect_toolchain()),
        )
        _decode_and_check(sample_prompt, "sampled-real-prompt", sample=True)
    except Exception as e:
        print(f"[sanity:sampled-real-prompt] could not build (toolchain?): {e}", flush=True)
        sample_prompt = (
            "<|im_start|>system\nYou are an AArch64 assembly writer.<|im_end|>\n"
            "<|im_start|>user\nWrite asm for `int kernel(int x){return x+1;}`.\n"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        _decode_and_check(sample_prompt, "sampled-fallback", sample=True)
    print("[sanity] PASS — model coherent on greedy AND sampled paths", flush=True)


def load_model(cfg: RunConfig, stack: str):
    print(f"[model] loading {cfg.model_id} stack={stack} rank={cfg.lora_rank} alpha={cfg.lora_alpha}", flush=True)

    if stack == "unsloth_vllm":
        from unsloth import FastLanguageModel
        model, tok = FastLanguageModel.from_pretrained(
            model_name=cfg.model_id,
            max_seq_length=cfg.max_prompt_length + cfg.max_completion_length,
            dtype=None,
            load_in_4bit=True,
            fast_inference=True,
        )
        if hasattr(tok, "truncation_side"):
            tok.truncation_side = "left"
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        _apply_qwen25_chat_eos_for_grpo(cfg.model_id, tok, model)
        try:
            from kaggle.apply_model_generate_stopping_shim import install_grpo_assembly_stopping
        except ImportError:
            from apply_model_generate_stopping_shim import install_grpo_assembly_stopping
        install_grpo_assembly_stopping(
            model,
            tok,
            min_new_tokens_before_close=cfg.assembly_min_new_tokens_before_close,
        )
        print(
            f"[model] unsloth loaded  eos_tok={getattr(tok, 'eos_token_id', None)}  "
            f"asm_min_new={cfg.assembly_min_new_tokens_before_close}  vram={_vram_str()}",
            flush=True,
        )
        return model, tok

    # bf16 LoRA (no 4-bit quantization).
    #
    # We previously used QLoRA via BitsAndBytesConfig(load_in_4bit=True). On the
    # HF-Jobs A10G image the bitsandbytes C++ extensions fail to load:
    #   "Skipping import of cpp extensions due to incompatible torch version.
    #    Please upgrade to torch >= 2.11.0 (found 2.10.0+cu128)."
    # The Python fallback for 4-bit dequantization corrupts logits, so the model
    # emits multilingual gibberish with low entropy regardless of the prompt
    # (every completion clipped to max_completion_length, never hits chat EOS).
    # e.g. 7B in bf16 ~14 GB, 14B ~28 GB -- use a GPU with enough headroom
    # for activations and GRPO rollouts, not just raw weight size.
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # GRPO truncates to max_prompt_length. Default is often right-truncation,
    # which cuts off the *tail* of the chat (where "Optimized Assembly Code:" lives).
    # Left-truncation keeps the anchor + end of the baseline assembly.
    tok.truncation_side = "left"

    # `eager` is the boring-correct path: no SDPA fused kernel, no flash-attn
    # JIT bind, no bnb-int8 hook surface. We previously saw multilingual
    # gibberish ('ampie', 'odzi', 'darm') in training rollouts even though the
    # standalone greedy sanity-check decoded cleanly. The pattern matches the
    # known "broken bitsandbytes CPP fallback on torch 2.10" failure mode
    # (bnb installs an int8 quantization patch on linear forwards even when
    # load_in_4bit=False; with mismatched torch ABI the patch corrupts logits).
    # `eager` + uninstalling bnb in the job script kills both surfaces.
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0" if torch.cuda.is_available() else "auto",
        attn_implementation="eager",
    )
    # bf16 LoRA prep: enable input grads so gradient checkpointing flows through
    # to the LoRA adapters (the kbit ``prepare`` helper did this for QLoRA).
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Explicit Qwen2 LoRA targets. Default PEFT auto-detection picks only q/v
    # which underfits asm; SuperCoder Appendix A.2 implies full attention+MLP
    # adapters for the 7B + assembly domain shift.
    lora = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.config.use_cache = False

    _sanity_check_inference(model, tok, cfg)

    _apply_qwen25_chat_eos_for_grpo(cfg.model_id, tok, model)
    _eos = getattr(model.generation_config, "eos_token_id", None) if model.generation_config else None
    try:
        from kaggle.apply_model_generate_stopping_shim import install_grpo_assembly_stopping
    except ImportError:
        from apply_model_generate_stopping_shim import install_grpo_assembly_stopping
    install_grpo_assembly_stopping(
        model,
        tok,
        min_new_tokens_before_close=cfg.assembly_min_new_tokens_before_close,
    )
    print(
        f"[model] bf16 loaded  eos_tok={getattr(tok, 'eos_token_id', None)}  gen_eos={_eos}  "
        f"asm_min_new={cfg.assembly_min_new_tokens_before_close}  vram={_vram_str()}",
        flush=True,
    )
    return model, tok


# ── GRPO config ───────────────────────────────────────────────────────────────

def grpo_config(cfg: RunConfig, stack: str) -> Any:
    from trl import GRPOConfig
    params: dict[str, Any] = dict(
        output_dir=cfg.out_dir,
        max_steps=cfg.steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        num_generations=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_grad_norm=0.1,
        temperature=cfg.temperature,
        # repetition_penalty: leave at TRL default 1.0. Values >1 break assembly:
        # mnemonics/registers repeat by design; HF's penalty up-weights rare tail
        # tokens → the exact "ampie"/"darm" gibberish seen in rollouts (see load_model
        # and _qwen25_grpo_generation_eos_kwargs comments). SuperCoder does not use this.
        # Divergent from teammate: pure GRPO with a small KL anchor instead
        # of DAPO's asymmetric clip. Different optimization regime explored
        # in parallel.
        loss_type="grpo",
        beta=0.04,
        epsilon=0.2,
        epsilon_high=0.28,
        # Truncation should not silently zero gradients. We bumped
        # max_completion_length to 1024; if a kernel still overflows, let the
        # partial (correctness/speedup=0) signal flow through to GRPO so the
        # policy at least learns "be shorter".
        mask_truncated_completions=False,
        remove_unused_columns=False,
        multi_objective_aggregation="normalize_then_sum",
        logging_steps=cfg.log_every,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        report_to="none",
    )
    gkw = _qwen25_grpo_generation_eos_kwargs(cfg)
    if gkw is not None:
        # Last-write wins in TRL’s internal GenerationConfig merge; fixes HF generate path
        params["generation_kwargs"] = gkw
    if stack == "unsloth_vllm":
        params.update(
            use_vllm=True,
            vllm_mode="colocate",
            vllm_enable_sleep_mode=True,
            vllm_importance_sampling_correction=True,
            vllm_importance_sampling_mode="sequence_truncate",
            vllm_importance_sampling_cap=2.0,
        )
    import re as _re
    while True:
        try:
            return GRPOConfig(**params)
        except TypeError as exc:
            m = _re.search(r"unexpected keyword argument '(\w+)'", str(exc))
            if not m:
                raise
            dropped = m.group(1)
            print(f"[grpo_config] dropping unsupported param: {dropped!r}", flush=True)
            params.pop(dropped, None)


# ── smoke test ────────────────────────────────────────────────────────────────

def smoke_only(cfg: RunConfig) -> int:
    tc = detect_toolchain()
    print(f"[smoke] toolchain clang={tc.clang} gcc={tc.gcc_aarch64} mca={tc.mca} mcpu={tc.mcpu}", flush=True)
    if not tc.ready() or not tc.mca:
        print("[smoke] FAIL — missing toolchain. re-run kaggle/setup.sh", flush=True)
        return 2
    dcfg = DatasetConfig(max_train=8, max_eval=2, difficulty_max=1)
    train, _, _ = build_dataset(tc, dcfg, tokenizer=None)
    print(f"[smoke] dataset rows: {len(train)}", flush=True)
    if len(train) == 0:
        print("[smoke] FAIL — dataset empty. check apt install", flush=True)
        return 2
    reward = LiveRewardFn.build()
    r = reward(
        completions=[f"<assembly>{train[0]['baseline_asm']}</assembly>"],
        variant_id=[train[0]["variant_id"]],
        baseline_asm=[train[0]["baseline_asm"]],
    )
    print(f"[smoke] reward for baseline-parity completion: {r}", flush=True)
    print("[smoke] OK", flush=True)
    return 0


# ── main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stack", default="auto",
                   choices=["auto", "unsloth_vllm", "plain_trl_ddp", "single_gpu"])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--out", "--out-dir", dest="out", default="runs/grpo-kaggle")
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Qwen2.5-Coder-7B default per hackathon wiki; 14B+ needs more VRAM.",
    )
    p.add_argument("--max-train", type=int, default=256)
    p.add_argument("--max-eval", type=int, default=32)
    p.add_argument("--difficulty-max", type=int, default=2)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--lora-dropout", type=float, default=None)
    p.add_argument(
        "--assembly-min-new-tokens-before-close",
        type=int,
        default=None,
        metavar="N",
        help="require N new tokens before </assembly> can stop generation (default: 64)",
    )
    p.add_argument(
        "--max-prompt-length",
        type=int,
        default=None,
        metavar="N",
        help="max prompt tokens for GRPO (default from RunConfig, usually 4096)",
    )
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    cfg_kwargs: dict[str, Any] = dict(
        model_id=args.model,
        stack=args.stack,
        steps=args.steps,
        lora_rank=args.lora_rank,
        num_generations=args.num_generations,
        out_dir=args.out,
        max_train=args.max_train,
        max_eval=args.max_eval,
        difficulty_max=args.difficulty_max,
        smoke=args.smoke,
    )
    if args.temperature is not None:
        cfg_kwargs["temperature"] = args.temperature
    if args.learning_rate is not None:
        cfg_kwargs["learning_rate"] = args.learning_rate
    if args.lora_dropout is not None:
        cfg_kwargs["lora_dropout"] = args.lora_dropout
    if args.assembly_min_new_tokens_before_close is not None:
        cfg_kwargs["assembly_min_new_tokens_before_close"] = (
            args.assembly_min_new_tokens_before_close
        )
    if args.max_prompt_length is not None:
        cfg_kwargs["max_prompt_length"] = args.max_prompt_length
    cfg = RunConfig(**cfg_kwargs)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print("=" * 60, flush=True)
    print(f"  ARM-Gym GRPO Training", flush=True)
    print(f"  model     : {cfg.model_id}", flush=True)
    print(f"  lora      : rank={cfg.lora_rank} alpha={cfg.lora_alpha} dropout={cfg.lora_dropout}", flush=True)
    print(
        f"  steps     : {cfg.steps}  lr={cfg.learning_rate}  temp={cfg.temperature}", flush=True,
    )
    print(
        f"  assembly  : min_new_tokens_before_close={cfg.assembly_min_new_tokens_before_close}",
        flush=True,
    )
    print(f"  out       : {cfg.out_dir}", flush=True)
    print("=" * 60, flush=True)

    if cfg.smoke:
        return smoke_only(cfg)

    stack = resolve_stack(cfg)
    print(f"[init] stack={stack}", flush=True)

    tc = detect_toolchain()
    if not tc.ready() or not tc.mca:
        print("[init] FAIL — toolchain missing. run kaggle/setup.sh first", file=sys.stderr)
        return 2
    print(f"[init] toolchain ok  mca={tc.mca}  mcpu={tc.mcpu}", flush=True)

    model, tok = load_model(cfg, stack)

    train_ds, eval_ds, _ = build_dataset(
        tc,
        DatasetConfig(max_train=cfg.max_train, max_eval=cfg.max_eval,
                      difficulty_max=cfg.difficulty_max),
        tokenizer=tok,
    )
    print(f"[init] dataset  train={len(train_ds)}  eval={len(eval_ds)}", flush=True)
    if len(train_ds) > 0:
        p0 = train_ds[0]["prompt"]
        tlen = len(tok(p0, add_special_tokens=True)["input_ids"])
        print(
            f"[init] first prompt: chars={len(p0)}  tok={tlen}  "
            f"max_prompt_length={cfg.max_prompt_length}  trunc_side={getattr(tok, 'truncation_side', '?')}",
            flush=True,
        )
        if tlen > cfg.max_prompt_length:
            print(
                f"[init] warn: first prompt > max_prompt_length — GRPO will truncate "
                f"(using side={getattr(tok, 'truncation_side', 'right')!r} on tokenizer)",
                flush=True,
            )
    if len(train_ds) == 0:
        print(
            "[init] FAIL — empty train dataset (every baseline compile failed). "
            "typical on HF: cross-gcc rejects -mcpu=neoverse-v3; "
            "arm_gym/compile_baseline now retries n1/v2/cortex-a76/generic. "
            "re-pull and retry, or run kaggle/setup.sh / apt install gcc-aarch64-linux-gnu",
            file=sys.stderr,
        )
        return 2

    gcfg = grpo_config(cfg, stack)

    logger = StepLogger(out / "log.csv", total_steps=cfg.steps)
    cb = make_callback(logger, total_steps=cfg.steps, cache_flush_every=20)

    from trl import GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward, syntax_reward, correctness_reward, speedup_reward],
        args=gcfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        callbacks=[cb],
    )

    print(f"\n[train] starting — {cfg.steps} steps  vram={_vram_str()}", flush=True)
    print("-" * 60, flush=True)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[train] interrupted by user — saving checkpoint", flush=True)
        trainer.save_model(str(out / "interrupted-checkpoint"))
    finally:
        logger.close()

        # final memory cleanup
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print("-" * 60, flush=True)
    print(f"[done] logs     -> {out}/log.csv", flush=True)
    print(f"[done] adapter  -> {out}/lora-adapter/", flush=True)
    print(f"[done] vram     -> {_vram_str()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
