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


# C + baseline stay in the row for the verifier; user message ends with an
# ``<assembly>`` prefill so the assistant continues inside the tag (Qwen chat).
SYSTEM_PROMPT = (
    "You write only AArch64 (aarch64-linux-gnu) assembly inside the user’s "
    "<assembly> block. No prose, no C, no other languages—assembly between "
    "the opening line the user started and a closing </assembly> tag."
)


def user_prompt(c_source: str, baseline_asm: str) -> str:
    return (
        "Generate ONLY AArch64 assembly.\n\n"
        "Wrap your continuation in <assembly></assembly> tags "
        "(the opening tag is already below—finish the block and close with "
        "</assembly>).\n\n"
        f"C code:\n{c_source}\n\n"
        f"Baseline assembly (gcc -O3):\n{baseline_asm}\n\n"
        "<assembly>\n"
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
    # Do not rstrip: user content must end with "<assembly>\n" for generation prefill.
    return prompt if prompt.endswith("\n") else prompt + "\n"


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
