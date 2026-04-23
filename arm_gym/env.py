"""OpenEnv-style environment + FastAPI server + curriculum.

Wiki free win 2 (curriculum-learning-rl): stage advancement gated on >80% pass
at difficulty d before unlocking d+1. Wiki free win 5 (structured errors):
every step returns a StructuredError payload the model can read in the next
generation.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from .compile_baseline import ToolchainInfo, compile_to_asm, detect_toolchain
from .errors import VerifierResult
from .kernels import KernelVariant, TEMPLATES, generate_all, split_train_eval
from .reward import RewardConfig, raw_reward
from .rollout_budget import TestCase
from .verifier import VerifierConfig, verify


class CompilerAction(BaseModel):
    variant_id: str
    assembly: str


class CompilerObservation(BaseModel):
    done: bool = False
    reward: float | None = None
    variant_id: str
    c_source: str
    baseline_asm: str
    baseline_cycles: float
    speedup: float | None = None
    error_json: str | None = None
    step_count: int = 0
    difficulty: int = 1


@dataclass
class Curriculum:
    stage: int = 1
    max_stage: int = 4
    pass_ema: float = 0.0
    threshold: float = 0.8
    alpha: float = 0.05

    def update(self, passed: bool) -> None:
        self.pass_ema = (1 - self.alpha) * self.pass_ema + self.alpha * (1.0 if passed else 0.0)
        if self.pass_ema > self.threshold and self.stage < self.max_stage:
            self.stage += 1
            self.pass_ema = 0.0

    def filter(self, variants: list[KernelVariant]) -> list[KernelVariant]:
        return [v for v in variants
                if TEMPLATES[v.template_name].difficulty <= self.stage]


@dataclass
class ARMGymEnv:
    toolchain: ToolchainInfo
    verifier_cfg: VerifierConfig
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)
    variants: list[KernelVariant] = field(default_factory=list)
    curriculum: Curriculum = field(default_factory=Curriculum)
    _baseline_cache: dict[str, tuple[str, float]] = field(default_factory=dict)
    _test_cache: dict[str, list[TestCase]] = field(default_factory=dict)

    @classmethod
    def build(cls, preferred_cpu: str = "neoverse-v3") -> "ARMGymEnv":
        tc = detect_toolchain(preferred_cpu)
        vcfg = VerifierConfig(
            mca_bin=tc.mca or "llvm-mca",
            assembler=os.environ.get("AARCH64_AS", "aarch64-linux-gnu-as"),
            linker=os.environ.get("AARCH64_LD", "aarch64-linux-gnu-ld"),
            qemu=os.environ.get("QEMU_AARCH64", "qemu-aarch64-static"),
            mcpu=tc.mcpu,
        )
        variants = generate_all()
        train, _ = split_train_eval(variants)
        return cls(toolchain=tc, verifier_cfg=vcfg, variants=train)

    def get_baseline(self, v: KernelVariant) -> tuple[str, float]:
        if v.variant_id in self._baseline_cache:
            return self._baseline_cache[v.variant_id]
        asm = compile_to_asm(v.c_source, self.toolchain)
        # cycle count proxy: use llvm-mca on baseline too for consistency
        try:
            from .mca import run_mca
            rep = run_mca(asm, self.verifier_cfg.mca_bin, self.verifier_cfg.mcpu)
            cycles = float(rep.total_cycles)
        except Exception:
            cycles = 1000.0  # pessimistic fallback so speedup still computable
        self._baseline_cache[v.variant_id] = (asm, cycles)
        return asm, cycles

    def tests_for(self, v: KernelVariant) -> list[TestCase]:
        # Placeholder: real implementation generates random inputs conforming
        # to the kernel signature + runs reference binary to compute expected.
        if v.variant_id not in self._test_cache:
            self._test_cache[v.variant_id] = []
        return self._test_cache[v.variant_id]

    def step(self, action: CompilerAction) -> CompilerObservation:
        v = next((x for x in self.variants if x.variant_id == action.variant_id), None)
        if v is None:
            return CompilerObservation(
                done=True, reward=0.0, variant_id=action.variant_id,
                c_source="", baseline_asm="", baseline_cycles=0.0,
                error_json='{"kind":"unknown_variant"}',
            )
        base_asm, base_cyc = self.get_baseline(v)
        result = verify(
            asm=action.assembly,
            baseline_asm=base_asm,
            variant_id=v.variant_id,
            tests=self.tests_for(v),
            cfg=self.verifier_cfg,
            run_correctness=lambda obj, t: True,  # wired in real run
            baseline_cycles=base_cyc,
        )
        self.curriculum.update(result.ok)
        r = raw_reward(result, self.reward_cfg, asm=action.assembly)
        return CompilerObservation(
            done=True,
            reward=r,
            variant_id=v.variant_id,
            c_source=v.c_source,
            baseline_asm=base_asm,
            baseline_cycles=base_cyc,
            speedup=result.speedup,
            error_json=result.error.to_prompt() if result.error else None,
            difficulty=self.curriculum.stage,
        )


app = FastAPI(title="arm-gym")


@app.get("/health")
def health() -> dict[str, Any]:
    tc = detect_toolchain()
    return {
        "ok": tc.ready(),
        "clang": tc.clang,
        "gcc_aarch64": tc.gcc_aarch64,
        "mca": tc.mca,
        "mcpu": tc.mcpu,
        "mcpu_disclosed": tc.mcpu_disclosed,
    }


_env_singleton: ARMGymEnv | None = None


def _env() -> ARMGymEnv:
    global _env_singleton
    if _env_singleton is None:
        _env_singleton = ARMGymEnv.build()
    return _env_singleton


@app.post("/step")
def step_endpoint(action: CompilerAction) -> CompilerObservation:
    return _env().step(action)
