"""Dataset builder: kernel variant → GRPO prompt.

One HF dataset row per variant, containing the C source, the baseline
assembly, and the variant_id. Reward function retrieves the env by variant_id
at step time.
"""

from __future__ import annotations

from dataclasses import dataclass
from datasets import Dataset

from arm_gym.compile_baseline import ToolchainInfo, compile_to_asm
from arm_gym.kernels import KernelVariant, generate_all, split_train_eval


# Prompt shape from SuperCoder Appendix A.3 (arXiv:2505.11480), ISA swapped
# to AArch64. Verbatim constraints: no extra text, no comments in asm, tags.
# PDF: meta-hackathon-llm-wiki/papers/supercoder.pdf
SYSTEM_PROMPT = (
    "You are an expert AArch64 (aarch64-linux-gnu-gcc) assembly writer. "
    "Obey the user block exactly. Output only what is asked in the required tags."
)


def user_prompt(c_source: str, baseline_asm: str) -> str:
    # Order and wording follow SuperCoder A.3 (x86-64 -> AArch64 for arm-gym).
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
        # `enable_thinking` is a Qwen3 flag; on Qwen2.5 templates it can be
        # quietly absorbed and break the rendered prompt for some tokenizer
        # versions. Try the strict (Qwen3) form first, then fall back to the
        # vanilla Qwen2.5 chat template.
        for kwargs in (
            {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False},
            {"tokenize": False, "add_generation_prompt": True},
        ):
            try:
                prompt = tokenizer.apply_chat_template(messages, **kwargs)
                break
            except TypeError:
                continue
        else:
            prompt = f"{SYSTEM_PROMPT}\n\n{messages[1]['content']}\n"
    else:
        prompt = f"{SYSTEM_PROMPT}\n\n{messages[1]['content']}\n"
    return prompt.rstrip()


@dataclass
class DatasetConfig:
    max_train: int = 256      # keep short — Kaggle run
    max_eval: int = 32
    difficulty_max: int = 2   # stage 1 + 2 only for first run


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
