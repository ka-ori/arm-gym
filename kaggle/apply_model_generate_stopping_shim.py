"""No-op stopping shim. Let the model terminate naturally on chat EOS.

Earlier versions installed a StoppingCriteria that suppressed termination until
``min_new_tokens_before_close`` tokens were generated, then stopped on
``</assembly>``. In practice this fought the chat EOS (``<|im_end|>``) and
forced the model to emit garbage padding, which collapsed the GRPO signal
(every completion clipped to exactly the floor length, ``reward_std=0``).

We now rely on:
  - ``<|im_end|>`` as the only stop token (set by ``_apply_qwen25_chat_eos_for_grpo``)
  - ``max_completion_length`` from GRPOConfig as the upper bound
  - The reward functions extracting ``<assembly>...</assembly>`` from completions
"""
from __future__ import annotations

from typing import Any


def install_grpo_assembly_stopping(
    model: Any,
    tokenizer: Any,
    min_new_tokens_before_close: int = 0,
    end_tag: str = "</assembly>",
) -> None:
    print(
        "[stopping-shim] DISABLED -- using natural chat EOS; "
        f"min_new_tokens_before_close={min_new_tokens_before_close} ignored",
        flush=True,
    )
    return
