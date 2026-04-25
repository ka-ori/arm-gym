"""Reward functions for GRPOTrainer - 3 independent callables for GDPO.

syntax_reward     : +1.0 if assembly parses, 0.0 otherwise
correctness_reward: +1.0 if QEMU executes without crash, 0.0 otherwise
speedup_reward    : (baseline_cycles / agent_cycles) - 1.0; 0.0 if !runs

Shared _Entry cache avoids 3x verifier calls per completion within a batch.
Cache is module-level and persistent (verifier is deterministic, results stable).
GDPO: pass [syntax_reward, correctness_reward, speedup_reward] to reward_funcs
and set multi_objective_aggregation="normalize_then_sum" in GRPOConfig.
"""

from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass

from arm_gym.compile_baseline import detect_toolchain
from arm_gym.verifier import run_correctness_qemu
from arm_gym.mca import run_mca
from arm_gym.rollout_budget import TestCase
from arm_gym.verifier import VerifierConfig, assemble, cleanup_temp_dirs

_ASM_RE = re.compile(r"<assembly>(.*?)</assembly>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_assembly(text: str) -> str:
    """Extract assembly body from text.

    The completion is expected to contain both ``<assembly>`` and ``</assembly>``
    tags wrapping the generated assembly code. We handle degraded cases too:
      1. Full ``<assembly>…</assembly>`` block → extract inner text.
      2. Only ``</assembly>`` → everything before the close tag is the body.
      3. Only ``<assembly>`` → everything after the open tag.
      4. Neither tag → treat the entire text as the body (best-effort).
    """
    text = _THINK_RE.sub("", text).strip()
    m = _ASM_RE.search(text)
    if m:
        return _clean_assembly(m.group(1))
    if "</assembly>" in text.lower():
        body = re.split(r"</assembly>", text, flags=re.IGNORECASE)[0]
        if "<assembly>" in body.lower():
            body = re.split(r"<assembly>", body, flags=re.IGNORECASE)[-1]
        return _clean_assembly(body)
    if "<assembly>" in text.lower():
        return _clean_assembly(re.split(r"<assembly>", text, flags=re.IGNORECASE)[-1])
    return _clean_assembly(text)


def _clean_assembly(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:asm|assembly|aarch64)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


@dataclass
class _Entry:
    assembles: bool = False
    runs: bool = False
    speedup: float = 0.0


_CACHE: dict[str, _Entry] = {}
_LOCK = threading.Lock()
_BASELINE: dict[str, float] = {}
_VCFG: VerifierConfig | None = None


def _cfg() -> VerifierConfig:
    global _VCFG
    if _VCFG is None:
        tc = detect_toolchain()
        _VCFG = VerifierConfig(
            mca_bin=tc.mca or "llvm-mca",
            assembler="aarch64-linux-gnu-as",
            linker="aarch64-linux-gnu-ld",
            qemu="qemu-aarch64-static",
            mcpu=tc.mcpu,
        )
    return _VCFG


def _bcy(vid: str, basm: str) -> float:
    if vid not in _BASELINE:
        try:
            rep = run_mca(basm, _cfg().mca_bin, _cfg().mcpu)
            _BASELINE[vid] = float(rep.total_cycles)
        except Exception:
            _BASELINE[vid] = 1000.0
    return _BASELINE[vid]


def _key(text: str, vid: str) -> str:
    return hashlib.md5(f"{text}::{vid}".encode()).hexdigest()


_DEBUG_LIMIT = 5
_DEBUG_SHOWN = 0


def _run(text: str, vid: str, basm: str) -> _Entry:
    """Compute (or retrieve cached) verify result for one completion."""
    global _DEBUG_SHOWN
    k = _key(text, vid)
    with _LOCK:
        if k in _CACHE:
            return _CACHE[k]

    # Compute outside lock - subprocess calls can be slow.
    # Two threads racing on the same key both compute and write; result is identical.
    e = _Entry()
    asm = extract_assembly(text)
    cfg = _cfg()

    # Gate 1: assemble
    obj, err = assemble(asm, cfg)
    if err or obj is None:
        if _DEBUG_SHOWN < _DEBUG_LIMIT:
            with _LOCK:
                if _DEBUG_SHOWN < _DEBUG_LIMIT:
                    _DEBUG_SHOWN += 1
                    print(
                        f"[reward-debug #{_DEBUG_SHOWN}] vid={vid} "
                        f"asm_fail={err.message[:200] if err else 'None'!r}",
                        flush=True,
                    )
                    print(f"[reward-debug] raw[:300]={text[:300]!r}", flush=True)
                    print(f"[reward-debug] extracted[:300]={asm[:300]!r}", flush=True)
        cleanup_temp_dirs()
        with _LOCK:
            _CACHE[k] = e
        return e
    e.assembles = True

    # Gate 2: QEMU run-without-crash
    e.runs = run_correctness_qemu(obj, TestCase(inputs=(), expected=None), cfg)

    # Gate 3: MCA speedup
    if e.runs:
        bc = _bcy(vid, basm)
        try:
            rep = run_mca(asm, cfg.mca_bin, cfg.mcpu)
            e.speedup = bc / max(rep.total_cycles, 1)
        except Exception:
            e.speedup = 0.0

    cleanup_temp_dirs()
    with _LOCK:
        _CACHE[k] = e
    return e


def _prep(completions, kwargs):
    texts = [c[-1]["content"] if isinstance(c, list) else str(c) for c in (completions or [])]
    n = len(texts)
    vids = list(kwargs.get("variant_id") or [""] * n)
    basms = list(kwargs.get("baseline_asm") or [""] * n)
    if len(vids) == 1 and n > 1:
        vids = vids * n
        basms = basms * n
    return texts, vids, basms


def syntax_reward(prompts=None, completions=None, **kwargs) -> list[float]:
    """Gate 1: +1.0 if assembly parses, 0.0 otherwise."""
    _ = prompts
    texts, vids, basms = _prep(completions, kwargs)
    return [1.0 if _run(t, v, b).assembles else 0.0 for t, v, b in zip(texts, vids, basms)]


def format_reward(prompts=None, completions=None, **kwargs) -> list[float]:
    """Shaping reward for well-formed completions.

    The model should emit both ``<assembly>`` and ``</assembly>`` tags.
      +0.1  has ``<assembly>`` open tag
      +0.1  has ``</assembly>`` close tag
      +0.1  has non-trivial assembly body (>= 20 non-whitespace chars between tags)
      -0.05 contains prose markers (code fences, think tags, explanations)
    Max score: 0.3 for a perfectly formed completion.
    """
    _ = prompts
    texts, _, _ = _prep(completions, kwargs)
    scores = []
    for text in texts:
        lowered = text.lower()
        has_open = "<assembly>" in lowered
        has_close = "</assembly>" in lowered
        body = extract_assembly(text)
        body_len = len(re.sub(r"\s", "", body))
        has_prose = any(marker in lowered
                        for marker in ("```", "<think>", "explain", "analysis"))
        score = 0.0
        if has_open:
            score += 0.1
        if has_close:
            score += 0.1
        if body_len >= 20:
            score += 0.1
        if has_prose:
            score -= 0.05
        scores.append(max(0.0, score))
    return scores


def correctness_reward(prompts=None, completions=None, **kwargs) -> list[float]:
    """Gate 2: +1.0 if QEMU executes without crash, 0.0 otherwise."""
    _ = prompts
    texts, vids, basms = _prep(completions, kwargs)
    return [1.0 if _run(t, v, b).runs else 0.0 for t, v, b in zip(texts, vids, basms)]


def speedup_reward(prompts=None, completions=None, **kwargs) -> list[float]:
    """Gate 3: max(0, baseline_cycles/agent_cycles - 1.0); 0.0 if assembly doesn't run.

    Clipped at 0 so slower-but-correct assembly never returns negative reward.
    Negative values suppressed z-score signal and discouraged correctness.
    """
    _ = prompts
    texts, vids, basms = _prep(completions, kwargs)
    scores = []
    for t, v, b in zip(texts, vids, basms):
        e = _run(t, v, b)
        scores.append(max(0.0, e.speedup - 1.0) if e.runs else 0.0)
    return scores


class LiveRewardFn:
    """Single-score wrapper kept for smoke test backward compat."""

    @classmethod
    def build(cls) -> "LiveRewardFn":
        return cls()

    def __call__(self, prompts=None, completions=None, **kwargs) -> list[float]:
        texts, vids, basms = _prep(completions, kwargs)
        out = []
        for t, v, b in zip(texts, vids, basms):
            e = _run(t, v, b)
            if not e.assembles:
                out.append(0.0)
            elif not e.runs:
                out.append(0.1)
            else:
                out.append(max(0.0, e.speedup - 1.0))
        return out
