"""Reward function — binary correctness + continuous clipped speedup.

Cut 3 fix: shaping terms DROPPED by default. Reward = 0 on any failure,
= clip(speedup - 1.0, 0, 2.0) on success. Secondary verifier catches hacking.
A `shaped` mode is available for ablation but requires liveness-aware NEON +
precise hazard def (see `mca.uses_neon_with_liveness` + `McaReport.no_pipeline_hazard`).

Cut 4 fix: z-score clip per group applied in `group_zscore_clip`. Apply this
on the batch of rewards produced for one prompt's group before GRPO advantage
computation. Bound: [-1.5, 1.5].
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Literal

from .errors import VerifierResult
from .mca import uses_neon_with_liveness


@dataclass
class RewardConfig:
    mode: Literal["binary_plus_speedup", "shaped"] = "binary_plus_speedup"
    speedup_clip_min: float = 1.0
    speedup_clip_max: float = 3.0
    zscore_clip: float = 1.5
    # shaped-mode only; wiki warns partial credit collapses K&K. Keep OFF by default.
    neon_bonus: float = 0.05
    no_hazard_bonus: float = 0.1


def raw_reward(v: VerifierResult, cfg: RewardConfig, asm: str | None = None) -> float:
    if not v.ok or v.speedup is None:
        return 0.0
    clipped = max(cfg.speedup_clip_min, min(v.speedup, cfg.speedup_clip_max))
    base = clipped - 1.0
    if cfg.mode == "binary_plus_speedup":
        return base
    # shaped: strict guards
    bonus = 0.0
    if asm is not None and uses_neon_with_liveness(asm):
        bonus += cfg.neon_bonus
    if v.mca_dispatch_stalls is not None and v.mca_resource_pressure_p99 is not None:
        if v.mca_dispatch_stalls == 0 and v.mca_resource_pressure_p99 < 1.0:
            bonus += cfg.no_hazard_bonus
    return base + bonus


def group_zscore_clip(rewards: list[float], clip: float = 1.5) -> list[float]:
    """Cut 4: z-score clip to [-clip, +clip] using group mean/std."""
    if len(rewards) < 2:
        return rewards
    mu = statistics.fmean(rewards)
    sd = statistics.pstdev(rewards) or 1e-6
    return [max(-clip, min(clip, (r - mu) / sd)) for r in rewards]


def batch_rewards(results: list[VerifierResult], cfg: RewardConfig,
                  asms: list[str] | None = None) -> list[float]:
    asms_iter = asms if asms is not None else [None] * len(results)
    raws = [raw_reward(v, cfg, a) for v, a in zip(results, asms_iter)]
    return group_zscore_clip(raws, clip=cfg.zscore_clip)
