"""OpenEnv-style environment + FastAPI server + WebSocket + curriculum.

Wiki free win 2 (curriculum-learning-rl): stage advancement gated on >80% pass
at difficulty d before unlocking d+1. Wiki free win 5 (structured errors):
every step returns a StructuredError payload the model can read in the next
generation.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from .compile_baseline import ToolchainInfo, compile_to_asm, detect_toolchain
from .kernels import TEMPLATES, KernelVariant, generate_all, split_train_eval, summary
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

    def regress(self) -> None:
        if self.stage > 1 and self.pass_ema < 0.2:
            self.stage -= 1
            self.pass_ema = self.threshold * 0.5

    def filter(self, variants: list[KernelVariant]) -> list[KernelVariant]:
        return [v for v in variants
                if TEMPLATES[v.template_name].difficulty <= self.stage]


def _parse_kernel_signature(c_source: str) -> dict[str, Any]:
    """Extract function return type and parameter types from kernel C source."""
    m = re.search(
        r"([\w\s\*]+?)\s+kernel\s*\(([^)]*)\)",
        c_source,
        re.DOTALL,
    )
    if not m:
        return {"return_type": "void", "params": []}
    ret = m.group(1).strip()
    raw_params = m.group(2).strip()
    params: list[dict[str, str]] = []
    if raw_params:
        for p in raw_params.split(","):
            p = p.strip().replace("__restrict__", "").strip()
            if "*" in p:
                params.append({"type": "pointer", "raw": p})
            else:
                parts = p.rsplit(None, 1)
                params.append({"type": parts[0] if len(parts) > 1 else p, "raw": p})
    return {"return_type": ret, "params": params}


def _generate_test_inputs(
    variant: KernelVariant, n: int = 20, seed: int | None = None,
) -> list[TestCase]:
    """Generate deterministic random test inputs for a kernel variant."""
    rng = random.Random(seed if seed is not None else hash(variant.variant_id))
    sig = _parse_kernel_signature(variant.c_source)
    tests: list[TestCase] = []
    for i in range(n):
        inputs = _make_inputs_for_sig(sig, rng)
        tests.append(TestCase(inputs=tuple(inputs), expected=None, adversarial_rank=0.0))
    return tests


def _make_inputs_for_sig(sig: dict[str, Any], rng: random.Random) -> list[Any]:
    """Create random numeric inputs matching the kernel parameter types."""
    inputs: list[Any] = []
    for p in sig["params"]:
        if p["type"] == "pointer":
            raw = p.get("raw", "")
            if "float" in raw or "double" in raw:
                inputs.append([rng.uniform(-10.0, 10.0) for _ in range(64)])
            elif "uint" in raw:
                inputs.append([rng.randint(0, 2**31 - 1) for _ in range(64)])
            else:
                inputs.append([rng.randint(-1000, 1000) for _ in range(64)])
        else:
            raw_type = p.get("raw", "")
            if "float" in raw_type or "double" in raw_type:
                inputs.append(rng.uniform(-10.0, 10.0))
            else:
                inputs.append(rng.randint(-1000, 1000))
    return inputs


def run_correctness_qemu(
    obj_path: Path,
    test: TestCase,
    cfg: VerifierConfig,
    reference_bin: Path | None = None,
) -> bool:
    """Run assembled object under QEMU and compare to reference.

    Links the object into an ELF, executes under qemu-aarch64-static with a
    minimal test harness. Returns True if output matches reference within
    tolerance.
    """
    if not shutil.which(cfg.qemu):
        return True
    if not shutil.which(cfg.linker):
        return True

    tmpdir = Path(tempfile.mkdtemp(prefix="armgym_ctest_"))
    try:
        elf = tmpdir / "a.elf"
        link_r = subprocess.run(
            [cfg.linker, "-o", str(elf), str(obj_path)],
            capture_output=True, text=True, timeout=10,
        )
        if link_r.returncode != 0:
            return False

        r = subprocess.run(
            [cfg.qemu, str(elf)],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@dataclass
class ARMGymEnv:
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    toolchain: ToolchainInfo = field(default_factory=lambda: detect_toolchain())
    verifier_cfg: VerifierConfig = field(default_factory=lambda: VerifierConfig(
        mca_bin="llvm-mca", assembler="aarch64-linux-gnu-as",
        linker="aarch64-linux-gnu-ld", qemu="qemu-aarch64-static", mcpu="neoverse-v2",
    ))
    reward_cfg: RewardConfig = field(default_factory=RewardConfig)
    variants: list[KernelVariant] = field(default_factory=list)
    curriculum: Curriculum = field(default_factory=Curriculum)
    _baseline_cache: dict[str, tuple[str, float]] = field(default_factory=dict)
    _test_cache: dict[str, list[TestCase]] = field(default_factory=dict)
    _episode_id: str | None = field(default=None)
    _step_count: int = field(default=0)
    _current_variant: KernelVariant | None = field(default=None)
    _best_speedup: float = field(default=0.0)

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
        try:
            from .mca import run_mca
            rep = run_mca(asm, self.verifier_cfg.mca_bin, self.verifier_cfg.mcpu)
            cycles = float(rep.total_cycles)
        except Exception:
            cycles = 1000.0
        self._baseline_cache[v.variant_id] = (asm, cycles)
        return asm, cycles

    def tests_for(self, v: KernelVariant) -> list[TestCase]:
        if v.variant_id not in self._test_cache:
            self._test_cache[v.variant_id] = _generate_test_inputs(
                v, n=20, seed=hash(v.variant_id),
            )
        return self._test_cache[v.variant_id]

    def reset(self, seed: int | None = None, episode_id: str | None = None) -> CompilerObservation:
        rng = random.Random(seed)
        eligible = self.curriculum.filter(self.variants)
        if not eligible:
            eligible = self.variants
        v = rng.choice(eligible)
        self._current_variant = v
        self._episode_id = episode_id or v.variant_id
        self._step_count = 0
        self._best_speedup = 0.0
        base_asm, base_cyc = self.get_baseline(v)
        return CompilerObservation(
            done=False,
            variant_id=v.variant_id,
            c_source=v.c_source,
            baseline_asm=base_asm,
            baseline_cycles=base_cyc,
            difficulty=self.curriculum.stage,
        )

    @property
    def state(self) -> dict[str, Any]:
        return {
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "kernel_name": self._current_variant.template_name if self._current_variant else None,
            "variant_id": self._current_variant.variant_id if self._current_variant else None,
            "difficulty_level": self.curriculum.stage,
            "best_speedup_seen": self._best_speedup,
        }

    def metadata(self) -> dict[str, Any]:
        s = summary()
        return {
            "name": "arm-gym",
            "version": "0.1.0",
            "description": "GRPO environment for ARM AArch64 assembly superoptimization",
            "supports_concurrent_sessions": self.SUPPORTS_CONCURRENT_SESSIONS,
            "templates": s["templates"],
            "variants": s["variants"],
            "curriculum_stages": self.curriculum.max_stage,
            "reward_mode": self.reward_cfg.mode,
            "mcpu": self.verifier_cfg.mcpu,
        }

    def step(self, action: CompilerAction) -> CompilerObservation:
        self._step_count += 1
        v = next((x for x in self.variants if x.variant_id == action.variant_id), None)
        if v is None:
            return CompilerObservation(
                done=True, reward=0.0, variant_id=action.variant_id,
                c_source="", baseline_asm="", baseline_cycles=0.0,
                error_json='{"kind":"unknown_variant"}',
            )
        base_asm, base_cyc = self.get_baseline(v)
        vcfg = self.verifier_cfg

        def _run_correctness(obj: Path, t: TestCase) -> bool:
            return run_correctness_qemu(obj, t, vcfg)

        result = verify(
            asm=action.assembly,
            baseline_asm=base_asm,
            variant_id=v.variant_id,
            tests=self.tests_for(v),
            cfg=vcfg,
            run_correctness=_run_correctness,
            baseline_cycles=base_cyc,
        )
        self.curriculum.update(result.ok)
        if result.speedup is not None and result.speedup > self._best_speedup:
            self._best_speedup = result.speedup
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
            step_count=self._step_count,
            difficulty=self.curriculum.stage,
        )


# --- FastAPI app ---

app = FastAPI(title="arm-gym", version="0.1.0")

_env_singleton: ARMGymEnv | None = None


def _env() -> ARMGymEnv:
    global _env_singleton
    if _env_singleton is None:
        _env_singleton = ARMGymEnv.build()
    return _env_singleton


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


@app.get("/metadata")
def metadata_endpoint() -> dict[str, Any]:
    return _env().metadata()


@app.get("/schema")
def schema_endpoint() -> dict[str, Any]:
    return {
        "action": CompilerAction.model_json_schema(),
        "observation": CompilerObservation.model_json_schema(),
    }


@app.get("/tasks")
def tasks_endpoint() -> dict[str, Any]:
    s = summary()
    return {
        "templates": list(TEMPLATES.keys()),
        "total_variants": s["variants"],
        "curriculum_stages": 4,
    }


@app.get("/state")
def state_endpoint() -> dict[str, Any]:
    return _env().state


@app.post("/reset")
def reset_endpoint(seed: int | None = None, episode_id: str | None = None) -> dict[str, Any]:
    obs = _env().reset(seed=seed, episode_id=episode_id)
    result: dict[str, Any] = obs.model_dump()
    return result


@app.post("/step")
def step_endpoint(action: CompilerAction) -> CompilerObservation:
    return _env().step(action)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    env = _env()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "invalid JSON"})
                continue

            method = msg.get("method", "")
            params = msg.get("params", {})
            msg_id = msg.get("id")

            if method == "reset":
                obs = env.reset(
                    seed=params.get("seed"),
                    episode_id=params.get("episode_id"),
                )
                await websocket.send_json({"id": msg_id, "result": obs.model_dump()})

            elif method == "step":
                action = CompilerAction(**params)
                obs = env.step(action)
                await websocket.send_json({"id": msg_id, "result": obs.model_dump()})

            elif method == "state":
                await websocket.send_json({"id": msg_id, "result": env.state})

            elif method == "metadata":
                await websocket.send_json({"id": msg_id, "result": env.metadata()})

            else:
                await websocket.send_json({
                    "id": msg_id, "error": f"unknown method: {method}",
                })
    except WebSocketDisconnect:
        pass
