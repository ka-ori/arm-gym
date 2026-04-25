"""Sanity check: Kaggle reward stack returns non-zero on a valid-looking completion."""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from kaggle.reward_fn import (  # noqa: E402
    correctness_reward,
    format_reward,
    speedup_reward,
    syntax_reward,
)

# Plausible AArch64: static Linux exit(0); assembles, links, qemu exit 0 when tools exist.
_ARM = """
.text
.global _start
_start:
  mov x8, #93
  mov x0, #0
  svc #0
"""


def _msg(content: str) -> list:
    return [[{"content": content}]]


def main() -> None:
    body = _ARM.strip()
    completion = f"<assembly>\n{body}\n</assembly>"
    kwargs = {"variant_id": "verify-rewards", "baseline_asm": body}
    comp = _msg(completion)
    fmt = format_reward(completions=comp, **kwargs)
    fmt_spread = format_reward(completions=comp + _msg("no closing tag"), **kwargs)
    st_fmt = statistics.pstdev(fmt_spread) if len(fmt_spread) > 1 else 0.0
    syn = cor = spd = [0.0]
    st_syn = 0.0
    try:
        syn = syntax_reward(completions=comp, **kwargs)
        cor = correctness_reward(completions=comp, **kwargs)
        spd = speedup_reward(completions=comp, **kwargs)
        st_syn = statistics.pstdev(
            syntax_reward(completions=comp + _msg("no assembly here"), **kwargs)
        ) or 0.0
    except (FileNotFoundError, OSError) as e:
        print("toolchain or qemu unavailable — syntax/correctness/speedup skipped:", e, flush=True)
    print("format", fmt, "syntax", syn, "correctness", cor, "speedup", spd)
    print("format_pstdev", st_fmt, "syntax_pstdev", st_syn)
    peak = max(fmt[0], syn[0], cor[0], spd[0])
    assert peak > 0 or st_fmt > 0 or st_syn > 0, "no reward signal"


if __name__ == "__main__":
    main()
