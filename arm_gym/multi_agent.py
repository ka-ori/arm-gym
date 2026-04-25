"""Analyzer + Optimizer with structured opportunity tokens.

Weaker flag 1 fix: instead of two prompts sharing one reward (role-prompting),
Analyzer emits a strict JSON opportunity-token list. Optimizer conditions on
those tokens. Per-role reward: Analyzer credit iff its suggestions appear in
winning Optimizer completion (token-match heuristic).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

OPPORTUNITY_KINDS = Literal[
    "vectorize_neon", "use_ldp_stp", "fuse_multiply_add", "replace_branch_with_csel",
    "unroll_loop", "eliminate_dead_stores", "use_sve2_predication",
    "reorder_for_dual_issue", "use_fmadd_fmsub", "hoist_constant",
]


@dataclass
class OpportunityToken:
    kind: str
    target_region: str | None = None  # e.g. "inner loop", "line 42"
    rationale: str | None = None

    def to_json(self) -> dict:
        return {"kind": self.kind, "target_region": self.target_region,
                "rationale": self.rationale}


ANALYZER_SYS_PROMPT = """You are an ARM AArch64 assembly ANALYZER.
Input: C source + baseline gcc -O3 assembly.
Output: strict JSON array of opportunity tokens only. No prose.
Schema: [{"kind": "<kind>", "target_region": "<string|null>", "rationale": "<string|null>"}, ...]
Allowed kinds: vectorize_neon, use_ldp_stp, fuse_multiply_add, replace_branch_with_csel,
  unroll_loop, eliminate_dead_stores, use_sve2_predication, reorder_for_dual_issue,
  use_fmadd_fmsub, hoist_constant.
Return at most 6 tokens. Empty array if no improvements available."""


OPTIMIZER_SYS_PROMPT = """You are an ARM AArch64 assembly OPTIMIZER.
Input: C source, baseline gcc -O3 assembly, and a JSON list of opportunity tokens.
Output: ONLY the optimized assembly inside <assembly></assembly> tags. No prose.
Preserve exact input/output semantics. Target: beat baseline on cycle count."""


def parse_analyzer_output(text: str) -> list[OpportunityToken]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        if isinstance(it, dict) and "kind" in it:
            out.append(OpportunityToken(
                kind=str(it.get("kind")),
                target_region=it.get("target_region"),
                rationale=it.get("rationale"),
            ))
    return out


def build_optimizer_prompt(c_source: str, baseline_asm: str,
                           tokens: list[OpportunityToken]) -> str:
    token_json = json.dumps([t.to_json() for t in tokens], indent=None)
    return f"""C Code:
{c_source}

Baseline Assembly (aarch64-linux-gnu-gcc -O3):
{baseline_asm}

Opportunity Tokens:
{token_json}

Wrap the optimized AArch64 assembly in <assembly></assembly>."""


# Token-match heuristic for analyzer credit.
_KIND_TO_MNEMONICS: dict[str, tuple[str, ...]] = {
    "vectorize_neon": ("ld1", "st1", "fmla", "add v", "sub v"),
    "use_ldp_stp": ("ldp", "stp"),
    "fuse_multiply_add": ("fmadd", "fmsub", "madd", "msub", "fmla"),
    "replace_branch_with_csel": ("csel", "csinc", "csinv"),
    "unroll_loop": (),  # detected via repeated block pattern, not mnemonic
    "eliminate_dead_stores": (),  # negative: fewer stores than baseline
    "use_sve2_predication": ("ptrue", "whilelo", "mov p"),
    "reorder_for_dual_issue": (),  # schedule-dependent
    "use_fmadd_fmsub": ("fmadd", "fmsub"),
    "hoist_constant": (),
}


def analyzer_credit(tokens: list[OpportunityToken], optimizer_asm: str) -> float:
    """Fraction of analyzer tokens whose mnemonics appear in the optimizer output."""
    if not tokens:
        return 0.0
    hits = 0
    counted = 0
    low = optimizer_asm.lower()
    for t in tokens:
        mnemonics = _KIND_TO_MNEMONICS.get(t.kind, ())
        if not mnemonics:
            continue
        counted += 1
        if any(m in low for m in mnemonics):
            hits += 1
    return hits / counted if counted else 0.0


def per_role_reward(optimizer_reward: float, tokens: list[OpportunityToken],
                    optimizer_asm: str) -> dict[str, float]:
    """Split reward so Analyzer is paid for useful guidance."""
    credit = analyzer_credit(tokens, optimizer_asm)
    # Analyzer gets a fraction of the optimizer's reward scaled by credit.
    return {
        "analyzer": optimizer_reward * 0.3 * credit,
        "optimizer": optimizer_reward,
    }
