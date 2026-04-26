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
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .compile_baseline import ToolchainInfo, compile_to_asm, detect_toolchain
from .kernels import TEMPLATES, KernelVariant, generate_all, split_train_eval, summary
from .reward import RewardConfig, raw_reward
from .rollout_budget import TestCase, make_edge_case_tests
from .verifier import VerifierConfig, assemble, run_correctness_qemu, verify


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
    """Stage-gated kernel sampling.

    Stage 0 = warm-start: only the simplest kernels at their smallest sizes
    (vec_add n=16, saxpy n=32, relu n=64). Once correctness EMA exceeds
    `warmstart_threshold` (default 0.6) we promote to stage 1 (all difficulty<=1
    templates), then continue the standard difficulty progression.
    """
    stage: int = 0
    max_stage: int = 4
    pass_ema: float = 0.0
    threshold: float = 0.8
    warmstart_threshold: float = 0.6
    alpha: float = 0.05

    # Warm-start whitelist: (template_name, n).
    WARMSTART_VARIANTS = (
        ("vec_add", 16),
        ("saxpy", 32),
        ("relu", 64),
    )

    def update(self, passed: bool) -> None:
        self.pass_ema = (1 - self.alpha) * self.pass_ema + self.alpha * (1.0 if passed else 0.0)
        # Warm-start → stage 1 only after correctness > warmstart_threshold.
        if self.stage == 0 and self.pass_ema > self.warmstart_threshold:
            self.stage = 1
            self.pass_ema = 0.0
            return
        if self.stage >= 1 and self.pass_ema > self.threshold and self.stage < self.max_stage:
            self.stage += 1
            self.pass_ema = 0.0

    def regress(self) -> None:
        if self.stage > 1 and self.pass_ema < 0.2:
            self.stage -= 1
            self.pass_ema = self.threshold * 0.5

    def filter(self, variants: list[KernelVariant]) -> list[KernelVariant]:
        if self.stage == 0:
            allowed: list[KernelVariant] = []
            for v in variants:
                if v.template_name not in {n for n, _ in self.WARMSTART_VARIANTS}:
                    continue
                target_n = next(
                    (n for name, n in self.WARMSTART_VARIANTS if name == v.template_name),
                    None,
                )
                params = dict(v.params)
                if target_n is not None and params.get("n") == target_n:
                    allowed.append(v)
            return allowed
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
    """Generate deterministic random test inputs for a kernel variant.

    EDGE_CASES (signed zeros, infinities, denormals, IEEE boundaries, int
    overflow edges) are mixed in with high adversarial_rank so the rolling
    top-N selection always exercises numerical-boundary failures first.
    """
    rng = random.Random(seed if seed is not None else hash(variant.variant_id))
    sig = _parse_kernel_signature(variant.c_source)
    tests: list[TestCase] = list(make_edge_case_tests())
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
    _reference_obj_cache: dict[str, Path] = field(default_factory=dict)
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

    def reference_obj(self, v: KernelVariant) -> Path | None:
        """Assemble baseline asm to a cached object file for output-comparison runs.

        Returns None if assembly of the reference itself fails (toolchain hiccup) —
        callers degrade to legacy exit-0 correctness.
        """
        if v.variant_id in self._reference_obj_cache:
            return self._reference_obj_cache[v.variant_id]
        base_asm, _ = self.get_baseline(v)
        obj, err = assemble(base_asm, self.verifier_cfg)
        if err or obj is None:
            return None
        self._reference_obj_cache[v.variant_id] = obj
        return obj

    def reset(self, seed: int | None = None, episode_id: str | None = None) -> CompilerObservation:
        rng = random.Random(seed)
        eligible = self.curriculum.filter(self.variants)
        # If warm-start whitelist matched nothing (e.g. dataset filtered out the
        # specific n=16/32/64 variants), promote to stage 1 once instead of
        # silently leaking higher-difficulty kernels.
        if not eligible and self.curriculum.stage == 0:
            self.curriculum.stage = 1
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
            # Stage 0 = warm-start (subset of difficulty-1 kernels). Report as 1
            # to keep the public observation field on the [1, max_stage] scale
            # consumers expect.
            difficulty=max(self.curriculum.stage, 1),
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
        ref_obj = self.reference_obj(v)

        def _run_correctness(obj: Path, t: TestCase) -> bool:
            return run_correctness_qemu(obj, t, vcfg, reference_bin=ref_obj)

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
            difficulty=max(self.curriculum.stage, 1),
        )


# --- FastAPI app ---

app = FastAPI(title="arm-gym", version="0.1.0")

_env_singleton: ARMGymEnv | None = None


def _env() -> ARMGymEnv:
    global _env_singleton
    if _env_singleton is None:
        _env_singleton = ARMGymEnv.build()
    return _env_singleton


_PLOTS = "https://huggingface.co/spaces/kaori02/arm-gym/resolve/main/eval/plots"
_NB    = "https://huggingface.co/spaces/kaori02/arm-gym/blob/main/eval/arm_gym_grpo_colab.ipynb"
_BLOG  = "https://huggingface.co/spaces/kaori02/arm-gym/blob/main/blog.md"
_MODEL = "https://huggingface.co/ZDC-M01/arm-gym-v11-train-250"

_INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ARM-Gym &mdash; AI vs Compiler</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#050505;--surface:#0a0a0a;--card:#0f0f0f;--card-up:#141414;
  --border:#1a1a1a;--border-hi:#242424;
  --text:#e8e6e3;--muted:#7d7870;--dim:#3e3a35;
  --gold:#c9a227;--gold-hi:#e2be45;--gold-lo:#a07d18;--gold-pale:#d4b94e;
  --gold-10:rgba(201,162,39,.10);--gold-05:rgba(201,162,39,.05);--gold-20:rgba(201,162,39,.20);
  --cream:#f0e4cc;
  --r:14px;
  --mono:'JetBrains Mono',ui-monospace,Menlo,monospace;
  --sans:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
}
html{scroll-behavior:smooth}
body{background:var(--bg);color:var(--text);font-family:var(--sans);line-height:1.6;overflow-x:hidden;-webkit-font-smoothing:antialiased}
a{color:inherit;text-decoration:none}
img{display:block;max-width:100%}

/* ── Animations ─────────────────────────────────────── */
@keyframes shimmer{0%{background-position:200% center}100%{background-position:-200% center}}
@keyframes glow-pulse{0%,100%{opacity:.5}50%{opacity:1}}
@keyframes slide-up{from{opacity:0;transform:translateY(40px)}to{opacity:1;transform:translateY(0)}}
@keyframes fade-in{from{opacity:0}to{opacity:1}}

.reveal{opacity:0;transform:translateY(30px);transition:opacity .7s cubic-bezier(.22,1,.36,1),transform .7s cubic-bezier(.22,1,.36,1)}
.reveal.visible{opacity:1;transform:translateY(0)}
.reveal-d1{transition-delay:.1s}.reveal-d2{transition-delay:.2s}
.reveal-d3{transition-delay:.3s}.reveal-d4{transition-delay:.4s}

/* ── Ambient Background ─────────────────────────────── */
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(ellipse 70% 45% at 50% 0%,rgba(201,162,39,.07),transparent),
    radial-gradient(ellipse 50% 40% at 80% 100%,rgba(201,162,39,.03),transparent);
}

/* ── Layout ─────────────────────────────────────────── */
.wrap{max-width:1080px;margin:0 auto;padding:0 1.5rem 2rem;position:relative;z-index:1}

/* ── Nav Bar ────────────────────────────────────────── */
.nav-outer{
  position:sticky;top:0;z-index:100;
  padding:.6rem 1.2rem 0;
  background:transparent;
}
.nav{
  max-width:820px;margin:0 auto;
  padding:0 .5rem 0 1.4rem;height:48px;
  display:flex;align-items:center;justify-content:space-between;
  background:rgba(15,15,15,.85);backdrop-filter:blur(20px) saturate(1.3);-webkit-backdrop-filter:blur(20px) saturate(1.3);
  border:1px solid var(--border-hi);border-radius:999px;
}
.nav-brand{
  font-family:var(--sans);font-size:1rem;font-weight:800;color:var(--gold);
  letter-spacing:.01em;
}
.nav-links{display:flex;gap:1.6rem;align-items:center}
.nav-links a{font-size:.82rem;font-weight:400;color:var(--muted);transition:color .2s}
.nav-links a:hover{color:var(--text)}
.nav-sep{width:1px;height:20px;background:var(--border-hi)}
.nav-cta{
  display:inline-flex;align-items:center;padding:.35rem .95rem;
  border-radius:999px;font-size:.78rem;font-weight:600;
  background:transparent;color:var(--text);
  border:1px solid var(--border-hi);
  transition:border-color .2s,color .2s;
}
.nav-cta:hover{border-color:var(--gold);color:var(--gold)}

/* ── Dividers ───────────────────────────────────────── */
.divider{display:flex;align-items:center;justify-content:center;gap:1rem;margin:1.5rem auto;max-width:200px}
.divider::before,.divider::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,transparent,var(--gold-lo),transparent)}
.divider-dot{width:6px;height:6px;border:1.5px solid var(--gold);transform:rotate(45deg)}

/* ── Hero ───────────────────────────────────────────── */
.hero{text-align:center;padding:1.5rem 0 1.5rem;position:relative}

.hero-badge{
  display:inline-block;font-size:.65rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  color:var(--gold);background:var(--gold-10);border:1px solid rgba(201,162,39,.25);
  padding:.3rem 1rem;border-radius:999px;margin-bottom:.8rem;
}

.hero-title{
  font-family:var(--sans);font-size:clamp(2rem,5vw,2.8rem);font-weight:800;
  letter-spacing:.02em;line-height:1.1;margin:.4rem 0 .5rem;
  color:var(--gold);
  text-shadow:0 0 12px rgba(201,162,39,.4),0 0 40px rgba(201,162,39,.15);
}
.hero-headline{
  font-size:clamp(1.1rem,2.5vw,1.5rem);font-weight:500;line-height:1.4;
  color:var(--cream);max-width:720px;margin:0 auto .5rem;
}
.hero-sub{font-size:.92rem;color:var(--muted);max-width:620px;margin:0 auto 1.2rem;line-height:1.6}

.tags{display:flex;gap:.5rem;justify-content:center;flex-wrap:wrap;margin-bottom:1.2rem}
.tag{
  font-size:.7rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
  padding:.3rem .85rem;border-radius:999px;transition:transform .2s,box-shadow .2s;
}
.tag:hover{transform:translateY(-1px)}
.tag-gold{background:var(--gold-10);border:1px solid var(--gold-20);color:var(--gold)}
.tag-outline{background:transparent;border:1px solid var(--border-hi);color:var(--muted)}

.cta-row{display:flex;gap:.65rem;justify-content:center;flex-wrap:wrap}
.cta{
  display:inline-flex;align-items:center;gap:.45rem;
  padding:.55rem 1.35rem;border-radius:10px;font-size:.85rem;font-weight:600;
  border:1px solid var(--border-hi);color:var(--text);background:var(--card);
  transition:all .3s cubic-bezier(.22,1,.36,1);position:relative;overflow:hidden;
}
.cta svg{width:14px;height:14px;opacity:.5;transition:opacity .3s}
.cta:hover svg{opacity:1}
.cta::after{
  content:'';position:absolute;inset:0;opacity:0;
  background:linear-gradient(135deg,rgba(201,162,39,.08),transparent 60%);
  transition:opacity .3s;
}
.cta:hover::after{opacity:1}
.cta:hover{border-color:var(--gold-lo);color:var(--gold-hi);transform:translateY(-2px);box-shadow:0 8px 30px rgba(0,0,0,.4)}
.cta-primary{
  border-color:var(--gold);color:var(--gold);
  background:linear-gradient(135deg,rgba(201,162,39,.1),rgba(201,162,39,.03));
}
.cta-primary:hover{background:var(--gold);color:#000;box-shadow:0 8px 35px rgba(201,162,39,.2)}
.cta-primary:hover svg{opacity:1;color:#000}

/* ── Section Headers ────────────────────────────────── */
.section{margin-bottom:0}
.s-head{margin-bottom:1.2rem}
.s-label{
  font-size:.7rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
  color:var(--gold);display:flex;align-items:center;gap:.75rem;margin-bottom:.5rem;
}
.s-label::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--gold-20),transparent 80%)}
.s-title{font-family:var(--sans);font-size:clamp(1.5rem,3.5vw,2rem);font-weight:800;color:var(--cream);letter-spacing:-.03em;line-height:1.25}
.s-sub{font-size:.92rem;color:var(--muted);margin-top:.4rem;max-width:580px;line-height:1.6}

/* ── Cards ──────────────────────────────────────────── */
.card-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:1rem}
@media(max-width:700px){.card-grid{grid-template-columns:1fr}}
.card{
  background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  padding:1.6rem;position:relative;overflow:hidden;
  transition:border-color .35s,transform .35s cubic-bezier(.22,1,.36,1),box-shadow .35s;
}
.card::before{
  content:'';position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent 10%,var(--gold-lo),transparent 90%);
  opacity:0;transition:opacity .35s;
}
.card:hover{border-color:var(--border-hi);transform:translateY(-3px);box-shadow:0 16px 48px rgba(0,0,0,.35)}
.card:hover::before{opacity:1}
.card-label{font-size:.65rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--gold);margin-bottom:.6rem}
.card h3{font-size:1.05rem;font-weight:700;line-height:1.35;color:var(--cream);margin-bottom:.5rem}
.card p{font-size:.87rem;color:var(--muted);line-height:1.65}
.card-num{
  position:absolute;top:1.2rem;right:1.4rem;
  font-family:var(--sans);font-size:2.5rem;font-weight:900;
  color:rgba(201,162,39,.06);line-height:1;pointer-events:none;
}

/* ── Stats ──────────────────────────────────────────── */
.metric-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;margin-bottom:1.2rem}
@media(max-width:700px){.metric-strip{grid-template-columns:repeat(2,1fr)}}
@media(max-width:480px){.metric-strip{grid-template-columns:1fr}}
.metric{
  background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  padding:1.4rem 1rem;text-align:center;position:relative;overflow:hidden;
  transition:border-color .3s,box-shadow .3s;
}
.metric::before{
  content:'';position:absolute;inset:0;opacity:0;
  background:radial-gradient(ellipse at 50% 0%,rgba(201,162,39,.06),transparent 70%);
  transition:opacity .4s;
}
.metric:hover{border-color:var(--border-hi)}
.metric:hover::before{opacity:1}
.metric .num{
  font-size:2.1rem;font-weight:800;font-family:var(--mono);line-height:1;
  background:linear-gradient(135deg,var(--gold-hi),var(--gold));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}
.metric .num.big{
  font-size:2.4rem;
  background:linear-gradient(135deg,var(--gold-hi),var(--gold),var(--gold-hi));
  background-size:300% auto;-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  animation:shimmer 4s linear infinite;
}
.metric .caption{font-size:.72rem;color:var(--muted);margin-top:.5rem;line-height:1.4;letter-spacing:.01em}

/* ── Plots ──────────────────────────────────────────── */
.plot-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem}
@media(max-width:700px){.plot-grid{grid-template-columns:1fr}}
.plot{
  background:var(--card);border:1px solid var(--border);border-radius:var(--r);overflow:hidden;
  transition:border-color .3s,box-shadow .3s;
}
.plot:hover{border-color:var(--border-hi);box-shadow:0 8px 30px rgba(0,0,0,.25)}
.plot img{width:100%;height:auto;display:block;object-fit:contain;background:var(--surface)}
.plot .placeholder{
  width:100%;padding:3rem 1rem;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:.5rem;
  background:var(--surface);color:var(--dim);font-size:.8rem;font-weight:500;
}
.plot .placeholder svg{width:32px;height:32px;opacity:.3}
.plot figcaption{font-size:.73rem;color:var(--muted);padding:.7rem .95rem;text-align:center;border-top:1px solid var(--border);min-height:3.2rem;display:flex;align-items:center;justify-content:center}

/* ── Mermaid Diagram ─────────────────────────────────── */
.mermaid-wrap{
  background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  padding:2rem 1.5rem;overflow-x:auto;position:relative;
}
.mermaid-wrap::before{
  content:'';position:absolute;top:0;left:10%;right:10%;height:1px;
  background:linear-gradient(90deg,transparent,var(--gold-lo),transparent);
}
.mermaid-wrap .mermaid{display:flex;justify-content:center}
.mermaid-wrap .mermaid svg{max-width:100%;height:auto}

/* ── API Table ──────────────────────────────────────── */
.api-wrap{background:var(--card);border:1px solid var(--border);border-radius:var(--r);overflow:hidden}
table{width:100%;border-collapse:collapse;font-size:.875rem}
thead{background:var(--gold-05)}
thead tr{border-bottom:1px solid var(--border)}
th{padding:.75rem 1rem;text-align:left;font-size:.65rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--gold-lo)}
td{padding:.7rem 1rem;border-bottom:1px solid var(--border);vertical-align:middle}
tbody tr{transition:background .2s}
tbody tr:last-child td{border-bottom:none}
tbody tr:hover{background:var(--gold-05)}
.badge{
  display:inline-block;font-family:var(--mono);font-size:.65rem;font-weight:700;
  border-radius:6px;padding:.18rem .5rem;min-width:2.8rem;text-align:center;letter-spacing:.03em;
}
.badge-get{background:var(--gold-10);color:var(--gold);border:1px solid var(--gold-20)}
.badge-post{background:rgba(226,190,69,.07);color:var(--gold-hi);border:1px solid rgba(226,190,69,.15)}
.badge-ws{background:rgba(160,125,24,.08);color:var(--gold-lo);border:1px solid rgba(160,125,24,.2)}
.ep{font-family:var(--mono);font-size:.82rem;color:var(--cream);transition:color .2s}
a.ep:hover{color:var(--gold)}
.ep-dim{font-family:var(--mono);font-size:.82rem;color:var(--muted)}
td .info{font-size:.82rem;color:var(--muted)}

/* ── Terminal Block ─────────────────────────────────── */
.terminal{
  background:var(--card);border:1px solid var(--border);border-radius:var(--r);
  overflow:hidden;
}
.terminal-bar{
  display:flex;align-items:center;gap:.5rem;padding:.6rem 1rem;
  background:var(--surface);border-bottom:1px solid var(--border);
}
.terminal-dot{width:10px;height:10px;border-radius:50%;border:1.5px solid var(--dim)}
.terminal-dot:first-child{border-color:#5a3a3a}.terminal-dot:nth-child(2){border-color:#5a5a3a}.terminal-dot:nth-child(3){border-color:#3a5a3a}
.terminal-title{font-size:.65rem;font-family:var(--mono);color:var(--dim);letter-spacing:.04em;margin-left:.3rem}
.terminal pre{border:none;border-radius:0;margin:0;background:transparent;padding:1.2rem 1.3rem}
.terminal pre::before{display:none}
.cmd{color:var(--cream)}.comment{color:var(--dim)}.str{color:var(--gold-pale)}


/* ── Footer ─────────────────────────────────────────── */
footer{
  text-align:center;padding:2rem 0 1.5rem;margin-top:1rem;
  border-top:1px solid var(--border);
}
.f-brand{font-family:var(--sans);font-size:.85rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:var(--gold);margin-bottom:.6rem}
.f-links{display:flex;gap:1.5rem;justify-content:center;flex-wrap:wrap;margin-bottom:.8rem}
.f-links a{font-size:.75rem;color:var(--muted);font-weight:500;transition:color .2s}
.f-links a:hover{color:var(--gold)}
.f-tagline{font-size:.72rem;color:var(--dim);margin-bottom:.3rem;line-height:1.5}
.f-note{font-size:.65rem;color:var(--dim);line-height:1.5}
</style>
</head>
<body>

<!-- ── NAV ──────────────────────────────────────────── -->
<div class="nav-outer">
<nav class="nav">
  <div class="nav-brand">ARM-Gym</div>
  <div class="nav-links">
    <a href="#results">Results</a>
    <a href="#how">Architecture</a>
    <a href="#api">API</a>
    <span class="nav-sep"></span>
    <a class="nav-cta" href="/docs" target="_blank" rel="noopener">API Docs</a>
  </div>
</nav>
</div>

<div class="wrap">

<!-- ── HERO ─────────────────────────────────────────── -->
<section class="hero">
  <h1 class="hero-title">ARM-Gym</h1>
  <p class="hero-headline">Can an AI write faster code than the world&rsquo;s best compiler?</p>
  <p class="hero-sub">Compilers translate your code into processor instructions. They are built to be safe for every program ever written. We trained an AI to find the faster instruction sequences the compiler won&rsquo;t try. The target: ARM, the architecture inside every smartphone, AWS data center, and AI chip.</p>
  <div class="cta-row">
    <a class="cta cta-primary" href="https://huggingface.co/spaces/kaori02/arm-gym/blob/main/blog.md" target="_blank" rel="noopener">
      <svg viewBox="0 0 20 20" fill="currentColor"><path d="M9 4.804A7.968 7.968 0 005.5 4c-1.255 0-2.443.29-3.5.804v10A7.969 7.969 0 015.5 14c1.669 0 3.218.51 4.5 1.385A7.962 7.962 0 0114.5 14c1.255 0 2.443.29 3.5.804v-10A7.968 7.968 0 0014.5 4c-1.255 0-2.443.29-3.5.804V14"/></svg>
      Read the Full Story
    </a>
    <a class="cta" href="https://huggingface.co/spaces/kaori02/arm-gym/blob/main/eval/arm_gym_grpo_colab.ipynb" target="_blank" rel="noopener">
      <svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M6 2a2 2 0 00-2 2v12a2 2 0 002 2h8a2 2 0 002-2V7.414A2 2 0 0015.414 6L12 2.586A2 2 0 0010.586 2H6z" clip-rule="evenodd"/></svg>
      Training Notebook
    </a>
    <a class="cta" href="https://huggingface.co/ZDC-M01/arm-gym-v11-train-250" target="_blank" rel="noopener">
      <svg viewBox="0 0 20 20" fill="currentColor"><path d="M13 7H7v6h6V7z"/><path fill-rule="evenodd" d="M7 2a1 1 0 012 0v1h2V2a1 1 0 112 0v1h2a2 2 0 012 2v2h1a1 1 0 110 2h-1v2h1a1 1 0 110 2h-1v2a2 2 0 01-2 2h-2v1a1 1 0 11-2 0v-1H9v1a1 1 0 11-2 0v-1H5a2 2 0 01-2-2v-2H2a1 1 0 110-2h1V9H2a1 1 0 010-2h1V5a2 2 0 012-2h2V2zM5 5h10v10H5V5z" clip-rule="evenodd"/></svg>
      Trained Model
    </a>
    <a class="cta" href="/docs" target="_blank" rel="noopener">
      <svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M12.316 3.051a1 1 0 01.633 1.265l-4 12a1 1 0 11-1.898-.632l4-12a1 1 0 011.265-.633zM5.707 6.293a1 1 0 010 1.414L3.414 10l2.293 2.293a1 1 0 11-1.414 1.414l-3-3a1 1 0 010-1.414l3-3a1 1 0 011.414 0zm8.586 0a1 1 0 011.414 0l3 3a1 1 0 010 1.414l-3 3a1 1 0 11-1.414-1.414L16.586 10l-2.293-2.293a1 1 0 010-1.414z" clip-rule="evenodd"/></svg>
      API Docs
    </a>
  </div>
</section>

<div class="divider"><span class="divider-dot"></span></div>

<!-- ── WHY ──────────────────────────────────────────── -->
<section class="section reveal">
  <div class="s-head">
    <div class="s-label">Why This Matters</div>
    <h2 class="s-title">Compilers play it safe.<br>We don&rsquo;t have to.</h2>
    <p class="s-sub">Every AI model you use runs on a processor. The code that drives that processor was written by a compiler. We&rsquo;re teaching an AI to write that code better than the compiler can.</p>
  </div>
  <div class="card-grid">
    <div class="card reveal reveal-d1">
      <span class="card-num">01</span>
      <div class="card-label">The Hardware</div>
      <h3>ARM is the world&rsquo;s most deployed processor.</h3>
      <p>Every smartphone, AWS Graviton cloud instances, Azure data centers, every Apple Mac since 2020, and Meta&rsquo;s in-house AI chips all run on ARM. Improving how code runs on ARM touches all of that.</p>
    </div>
    <div class="card reveal reveal-d2">
      <span class="card-num">02</span>
      <div class="card-label">The Problem</div>
      <h3>Compilers are brilliant generalists with a blind spot.</h3>
      <p>A compiler translates your code into processor instructions. It is optimized to be safe for every possible program ever written. That safety comes at a cost: on specific hardware, for specific workloads, there are faster instruction sequences the compiler will never try because it cannot afford to be wrong even once.</p>
    </div>
    <div class="card reveal reveal-d3">
      <span class="card-num">03</span>
      <div class="card-label">The Idea</div>
      <h3>A language model as a probabilistic scout.</h3>
      <p>ARM-Gym gives a 7-billion-parameter AI a programming function and asks it to rewrite the processor instructions from scratch. Every attempt is verified by a real assembler, a hardware emulator, and a cycle counter. The AI explores. The verifier confirms.</p>
    </div>
    <div class="card reveal reveal-d4">
      <span class="card-num">04</span>
      <div class="card-label">The Result</div>
      <h3>The model learned to write valid ARM assembly, then started beating the compiler.</h3>
      <p>At the start of training, the AI&rsquo;s assembly was correct only 19% of the time. By the end of 250 steps, it was correct 70% of the time, improving every quarter. Once it learned to write assembly that actually runs, it started finding sequences faster than the compiler. No prior system has done this on ARM.</p>
    </div>
  </div>
</section>

<div class="divider"><span class="divider-dot"></span></div>

<!-- ── RESULTS ──────────────────────────────────────── -->
<section class="section reveal" id="results">
  <div class="s-head">
    <div class="s-label">Training Results</div>
    <h2 class="s-title">250 Steps on an NVIDIA L40S</h2>
    <p class="s-sub">Qwen2.5-Coder-7B-Instruct with LoRA fine-tuning, trained via GRPO on 649 kernel variants. 107 minutes on a single GPU.</p>
  </div>
  <div class="metric-strip reveal">
    <div class="metric">
      <div class="num big">70%</div>
      <div class="caption">Assembly Correctness at End<br>Started at 19% &mdash; rose every quarter</div>
    </div>
    <div class="metric">
      <div class="num">6.50</div>
      <div class="caption">Final Quarter Reward<br>Up from 3.30 at the start</div>
    </div>
    <div class="metric">
      <div class="num">+14.5%</div>
      <div class="caption">Best Speedup Over Compiler<br>Cycle estimate, single best event</div>
    </div>
    <div class="metric">
      <div class="num big">649</div>
      <div class="caption">Kernel Variants Trained On<br>15 AI inference templates</div>
    </div>
    <div class="metric">
      <div class="num">250</div>
      <div class="caption">Training Steps<br>107 minutes wall clock</div>
    </div>
    <div class="metric">
      <div class="num">&lt;1ms</div>
      <div class="caption">Per Attempt Verification<br>Deterministic, no runtime noise</div>
    </div>
  </div>
  <div class="plot-grid reveal">
    <figure class="plot">
      <img src="https://huggingface.co/spaces/kaori02/arm-gym/resolve/main/eval/plots/fig2_rewards.png" alt="Reward components across 250 training steps" loading="lazy">
      <figcaption>All four reward signals across 250 training steps: format, syntax, correctness, and speedup</figcaption>
    </figure>
    <figure class="plot">
      <img src="https://huggingface.co/spaces/kaori02/arm-gym/resolve/main/eval/plots/fig3_correctness.png" alt="QEMU correctness rate rising from 19% to 70%" loading="lazy">
      <figcaption>Assembly correctness verified by running 20 randomized tests per attempt. Rose from 19% to 70% across 250 steps, improving every quarter.</figcaption>
    </figure>
    <figure class="plot">
      <img src="https://huggingface.co/spaces/kaori02/arm-gym/resolve/main/eval/plots/fig4_comparison.png" alt="Speedup events and V10 vs V11 comparison" loading="lazy">
      <figcaption>Steps where the model beat the compiler on cycle estimates, and a comparison between two training runs showing the improvement.</figcaption>
    </figure>
    <figure class="plot">
      <img src="https://huggingface.co/spaces/kaori02/arm-gym/resolve/main/eval/plots/fig5_trajectory.png" alt="Total reward trajectory Q1=3.30 to Q4=6.50" loading="lazy">
      <figcaption>Total reward each quarter: 3.30, 4.20, 5.40, 6.50. Every quarter stronger than the last, compared against the previous training run.</figcaption>
    </figure>
  </div>
</section>

<div class="divider"><span class="divider-dot"></span></div>

<!-- ── HOW IT WORKS ─────────────────────────────────── -->
<section class="section reveal" id="how">
  <div class="s-head">
    <div class="s-label">Architecture</div>
    <h2 class="s-title">How It Works</h2>
    <p class="s-sub">A three-gate verification pipeline. Zero LLM judges. Fully deterministic reward.</p>
  </div>
  <div class="card-grid" style="margin-bottom:1.2rem">
    <div class="card reveal reveal-d1">
      <span class="card-num">A</span>
      <div class="card-label">The Environment</div>
      <h3>C function in, ARM assembly out.</h3>
      <p>Each step selects a C kernel from 649 variants across 15 templates. The compiler baseline is generated with clang-21 -O3. The model receives both and writes optimized AArch64 assembly.</p>
    </div>
    <div class="card reveal reveal-d2">
      <span class="card-num">B</span>
      <div class="card-label">The Verification</div>
      <h3>Three gates. No AI judge.</h3>
      <p>Every attempt must pass: a real assembler for syntax, a hardware simulator running 20 adversarial tests for correctness, and a cycle counter for performance.</p>
    </div>
  </div>
  <div class="mermaid-wrap reveal">
    <pre class="mermaid">
flowchart LR
    A["C Kernel&lt;br/&gt;15 templates x 649 variants"] --> B["clang-21 -O3&lt;br/&gt;Baseline Assembly"]
    B --> C["LLM Prompt&lt;br/&gt;C + Baseline ASM"]
    C --> D["Qwen2.5-Coder-7B&lt;br/&gt;+ LoRA r=32 G=8"]
    D --> E["Agent Assembly"]
    E --> F{"3-Gate&lt;br/&gt;Verifier"}
    F -->|"Syntax"| G["GNU as&lt;br/&gt;aarch64"]
    F -->|"Correctness"| H["QEMU x 20&lt;br/&gt;Adversarial Tests"]
    F -->|"Performance"| I["LLVM-MCA&lt;br/&gt;Neoverse V2"]
    I --> J["Dual Verifier&lt;br/&gt;Cross-Check"]
    J --> K["Reward&lt;br/&gt;fmt+syntax+correct+speedup"]
    K --> L["GRPO&lt;br/&gt;z-score clip ±1.5"]
    L --> D
    </pre>
  </div>
</section>

<div class="divider"><span class="divider-dot"></span></div>

<!-- ── TRAINING SCRIPTS ─────────────────────────────── -->
<section class="section reveal">
  <div class="s-head">
    <div class="s-label">Reproduce the Training</div>
    <h2 class="s-title">Everything is open.</h2>
    <p class="s-sub">Both training runs are fully reproducible. The scripts, logs, and trained weights are all public.</p>
  </div>
  <div class="card-grid">
    <div class="card reveal reveal-d1">
      <span class="card-num">V11</span>
      <div class="card-label">Primary Run</div>
      <h3>Qwen2.5-Coder-7B + LoRA r=32, 250 steps</h3>
      <p>107 minutes on a single NVIDIA L40S. 8 generations per step. 649 kernel variants. This is the run all results are based on.</p>
      <p style="margin-top:.8rem">
        <a href="https://huggingface.co/spaces/kaori02/arm-gym/blob/main/hf/v11_train.py" target="_blank" rel="noopener" style="color:var(--gold);text-decoration:none;font-size:.85rem">Training script (v11_train.py) &rarr;</a><br>
        <a href="https://huggingface.co/ZDC-M01/arm-gym-v11-train-250" target="_blank" rel="noopener" style="color:var(--gold);text-decoration:none;font-size:.85rem">Trained LoRA adapters &rarr;</a>
      </p>
    </div>
    <div class="card reveal reveal-d2">
      <span class="card-num">V10</span>
      <div class="card-label">Comparison Run</div>
      <h3>Qwen2.5-Coder-7B + LoRA r=24, 200 steps</h3>
      <p>94 minutes on a single NVIDIA L40S. 6 generations per step. Used as the baseline comparison in all plots. Shows what a slightly smaller config achieves.</p>
      <p style="margin-top:.8rem">
        <a href="https://huggingface.co/spaces/kaori02/arm-gym/blob/main/hf/v10_train.py" target="_blank" rel="noopener" style="color:var(--gold);text-decoration:none;font-size:.85rem">Training script (v10_train.py) &rarr;</a><br>
        <a href="https://huggingface.co/spaces/kaori02/arm-gym/resolve/main/logs/arm-gym-logs.zip" target="_blank" rel="noopener" style="color:var(--gold);text-decoration:none;font-size:.85rem">Training logs, both runs (CSV) &rarr;</a>
      </p>
    </div>
    <div class="card reveal reveal-d3">
      <span class="card-num">NB</span>
      <div class="card-label">Colab Notebook</div>
      <h3>Key training steps with dependency notes</h3>
      <p>The notebook was built to run on HuggingFace infrastructure. It has dependency notes for running on Colab and snippets showing each stage of the GRPO training loop.</p>
      <p style="margin-top:.8rem">
        <a href="https://huggingface.co/spaces/kaori02/arm-gym/blob/main/eval/arm_gym_grpo_colab.ipynb" target="_blank" rel="noopener" style="color:var(--gold);text-decoration:none;font-size:.85rem">Open notebook &rarr;</a>
      </p>
    </div>
  </div>
</section>

<div class="divider"><span class="divider-dot"></span></div>

<!-- ── TRY IT ────────────────────────────────────────── -->
<section class="section reveal" id="api">
  <div class="s-head">
    <div class="s-label">Live Environment</div>
    <h2 class="s-title">Try it now.</h2>
    <p class="s-sub">The environment is running. Hit <a href="/docs" target="_blank" style="color:var(--gold)">interactive Swagger docs</a> to test every endpoint directly in the browser, or use curl.</p>
  </div>
  <div style="text-align:center;margin-bottom:2rem">
    <a class="cta cta-primary" href="/docs" target="_blank" rel="noopener" style="display:inline-flex">
      <svg viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M12.316 3.051a1 1 0 01.633 1.265l-4 12a1 1 0 11-1.898-.632l4-12a1 1 0 011.265-.633zM5.707 6.293a1 1 0 010 1.414L3.414 10l2.293 2.293a1 1 0 11-1.414 1.414l-3-3a1 1 0 010-1.414l3-3a1 1 0 011.414 0zm8.586 0a1 1 0 011.414 0l3 3a1 1 0 010 1.414l-3 3a1 1 0 11-1.414-1.414L16.586 10l-2.293-2.293a1 1 0 010-1.414z" clip-rule="evenodd"/></svg>
      Open Interactive API Docs
    </a>
  </div>
  <div class="terminal">
    <div class="terminal-bar">
      <span class="terminal-dot"></span><span class="terminal-dot"></span><span class="terminal-dot"></span>
      <span class="terminal-title">bash</span>
    </div>
<pre><span class="comment"># Check toolchain: gcc, llvm-mca, QEMU</span>
<span class="cmd">curl</span> -s <span class="str">"https://kaori02-arm-gym.hf.space/health"</span>

<span class="comment"># Get a kernel to optimize (C source + baseline assembly + cycle count)</span>
<span class="cmd">curl</span> -s -X POST <span class="str">"https://kaori02-arm-gym.hf.space/reset?seed=42"</span>

<span class="comment"># Submit assembly, get reward (syntax + correctness + speedup)</span>
<span class="cmd">curl</span> -s -X POST <span class="str">"https://kaori02-arm-gym.hf.space/step"</span> \\
  -H <span class="str">"content-type: application/json"</span> \\
  -d <span class="str">'{"variant_id":"vec_add_n16_float32","assembly":".text\n.global kernel\nkernel:\n  ret"}'</span>

<span class="comment"># All 649 kernel variants by difficulty</span>
<span class="cmd">curl</span> -s <span class="str">"https://kaori02-arm-gym.hf.space/tasks"</span></pre>
  </div>
</section>

<!-- ── FOOTER ───────────────────────────────────────── -->
<footer>
  <div class="f-brand">ARM-Gym</div>
  <div class="f-links">
    <a href="https://huggingface.co/ZDC-M01/arm-gym-v11-train-250" target="_blank" rel="noopener">Trained LoRA</a>
    <a href="https://huggingface.co/spaces/kaori02/arm-gym/blob/main/blog.md" target="_blank" rel="noopener">Blog</a>
    <a href="https://huggingface.co/spaces/kaori02/arm-gym/blob/main/eval/arm_gym_grpo_colab.ipynb" target="_blank" rel="noopener">Notebook</a>
    <a href="/docs" target="_blank" rel="noopener">API Docs</a>
  </div>
  <div class="f-tagline">Meta / HuggingFace OpenEnv Hackathon India 2026 &mdash; Team (dot)mkv</div>
  <div class="f-note">All speedup figures are LLVM-MCA model estimates on Neoverse V2. Silicon validation pending.</div>
</footer>

</div>

<script type="module">
import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
mermaid.initialize({
  startOnLoad: true,
  theme: 'base',
  themeVariables: {
    darkMode: true,
    background: '#0f0f0f',
    primaryColor: '#1a1508',
    primaryTextColor: '#e8e6e3',
    primaryBorderColor: '#c9a227',
    secondaryColor: '#141008',
    secondaryTextColor: '#e8e6e3',
    secondaryBorderColor: '#a07d18',
    tertiaryColor: '#0f0f0f',
    tertiaryTextColor: '#e8e6e3',
    tertiaryBorderColor: '#3e3a35',
    lineColor: '#c9a227',
    textColor: '#e8e6e3',
    mainBkg: '#1a1508',
    nodeBorder: '#c9a227',
    clusterBkg: '#0a0a0a',
    clusterBorder: '#3e3a35',
    titleColor: '#c9a227',
    edgeLabelBackground: '#0f0f0f',
    nodeTextColor: '#e8e6e3',
    actorTextColor: '#e8e6e3',
    labelTextColor: '#d4b94e',
    loopTextColor: '#e8e6e3',
    noteBkgColor: '#1a1508',
    noteTextColor: '#e8e6e3',
    noteBorderColor: '#c9a227',
    fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
    fontSize: '13px'
  },
  flowchart: {
    htmlLabels: true,
    curve: 'basis',
    padding: 16,
    nodeSpacing: 30,
    rankSpacing: 50,
    useMaxWidth: true
  }
});
</script>
<script>
(function(){
  var io=new IntersectionObserver(function(entries){
    entries.forEach(function(e){if(e.isIntersecting){e.target.classList.add('visible');io.unobserve(e.target)}})
  },{threshold:0.12,rootMargin:'0px 0px -40px 0px'});
  document.querySelectorAll('.reveal').forEach(function(el){io.observe(el)});
})();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _INDEX_HTML


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
    try:
        obs = _env().reset(seed=seed, episode_id=episode_id)
        return obs.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step")
def step_endpoint(action: CompilerAction) -> CompilerObservation:
    try:
        return _env().step(action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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
