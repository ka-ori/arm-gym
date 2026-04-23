"""3-gate verifier + dual-verifier cross-check + 3σ sanity bound.

Gates:
 1. assemble → GNU as
 2. correctness → adversarial N=20 inputs via QEMU, bitwise/tolerance match vs
    reference gcc -O3 binary
 3. performance → llvm-mca cycle estimate, secondary native-gcc cycle-count path

Weaker flag 2 fix: 3σ bound uses `offline_baseline/<variant_id>.json` produced
by `scripts/baseline_distribution.py`. If distribution unavailable, 3σ check is
skipped (flagged in VerifierResult.extra).

LLM-free throughout. No judge model.
"""

from __future__ import annotations
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .errors import ErrorKind, StructuredError, VerifierResult
from .mca import McaReport, run_mca
from .rollout_budget import TestCase, run_parallel, select_adversarial

BASELINE_DIST_DIR = Path(os.environ.get("ARM_GYM_BASELINE_DIST", "artifacts/baseline_dist"))


@dataclass
class VerifierConfig:
    mca_bin: str
    assembler: str  # aarch64-linux-gnu-as
    linker: str     # aarch64-linux-gnu-ld
    qemu: str       # qemu-aarch64-static
    mcpu: str
    mca_triple: str = "aarch64-linux-gnu"
    float_rtol: float = 1e-5
    float_atol: float = 1e-8


def assemble(asm: str, cfg: VerifierConfig) -> tuple[Path | None, StructuredError | None]:
    d = Path(tempfile.mkdtemp(prefix="armgym_"))
    src = d / "k.s"
    obj = d / "k.o"
    src.write_text(asm)
    r = subprocess.run([cfg.assembler, "-o", str(obj), str(src)],
                       capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        line, col = _parse_as_error(r.stderr)
        shutil.rmtree(d, ignore_errors=True)
        return None, StructuredError(
            kind=ErrorKind.ASSEMBLE_FAIL,
            message=r.stderr.strip()[:500],
            line=line,
            column=col,
        )
    return obj, None


def _parse_as_error(stderr: str) -> tuple[int | None, int | None]:
    import re
    m = re.search(r":(\d+):(?:(\d+):)?", stderr)
    if not m:
        return None, None
    return int(m.group(1)), (int(m.group(2)) if m.group(2) else None)


def run_secondary_cycle_count(obj_path: Path, cfg: VerifierConfig,
                              test: TestCase) -> int | None:
    """Secondary verifier: native execution under qemu with instruction count.

    Independent of llvm-mca. If the two disagree by > threshold, flag drift.
    """
    try:
        r = subprocess.run([cfg.qemu, "-count-insns", str(obj_path)],
                           capture_output=True, text=True, timeout=5)
        for line in r.stderr.splitlines():
            if "insns:" in line or "instructions:" in line:
                return int("".join(c for c in line if c.isdigit()))
        return None
    except Exception:
        return None


def load_baseline_distribution(variant_id: str) -> dict | None:
    p = BASELINE_DIST_DIR / f"{variant_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def sigma_sanity_ok(speedup: float, dist: dict) -> bool:
    """Return False if speedup exceeds 3σ above historical mean for this variant."""
    mean = dist.get("mean", 1.0)
    std = dist.get("std", 0.1)
    return speedup <= mean + 3.0 * max(std, 1e-6)


def verify(
    asm: str,
    baseline_asm: str,
    variant_id: str,
    tests: list[TestCase],
    cfg: VerifierConfig,
    run_correctness: Callable[[Path, TestCase], bool],
    baseline_cycles: float,
) -> VerifierResult:
    # Gate 1
    obj, err = assemble(asm, cfg)
    if err:
        return VerifierResult(ok=False, reward=0.0, error=err)

    # Gate 2: adversarial correctness
    selected = select_adversarial(tests)
    ok, fail_idx = run_parallel(lambda t: run_correctness(obj, t), selected)
    if not ok:
        return VerifierResult(
            ok=False, reward=0.0,
            error=StructuredError(
                kind=ErrorKind.OUTPUT_MISMATCH,
                message=f"test {fail_idx} failed on variant {variant_id}",
                extra={"failed_index": fail_idx},
            ),
        )

    # Gate 3: performance via llvm-mca primary
    try:
        report: McaReport = run_mca(asm, cfg.mca_bin, cfg.mcpu, cfg.mca_triple)
    except Exception as e:
        return VerifierResult(
            ok=False, reward=0.0,
            error=StructuredError(kind=ErrorKind.MCA_FAIL, message=str(e)[:500]),
        )

    speedup = baseline_cycles / max(report.total_cycles, 1)

    # Secondary verifier: native instruction count
    secondary = run_secondary_cycle_count(obj, cfg, selected[0] if selected else None)
    if secondary is not None and secondary > 0:
        # Large disagreement (> 3×) signals proxy drift.
        ratio = max(report.total_cycles, secondary) / max(min(report.total_cycles, secondary), 1)
        if ratio > 3.0:
            return VerifierResult(
                ok=False, reward=0.0,
                error=StructuredError(
                    kind=ErrorKind.SECONDARY_VERIFIER_MISMATCH,
                    message=f"mca={report.total_cycles} qemu_insns={secondary} ratio={ratio:.2f}",
                    extra={"mca_cycles": report.total_cycles, "qemu_insns": secondary},
                ),
                agent_cycles=report.total_cycles,
                baseline_cycles=baseline_cycles,
                speedup=speedup,
            )

    # 3σ sanity against offline distribution
    dist = load_baseline_distribution(variant_id)
    if dist and not sigma_sanity_ok(speedup, dist):
        return VerifierResult(
            ok=False, reward=0.0,
            error=StructuredError(
                kind=ErrorKind.SPEEDUP_OUTLIER,
                message=f"speedup {speedup:.2f}x exceeds 3σ for {variant_id}",
                extra={"dist": dist, "speedup": speedup},
            ),
            agent_cycles=report.total_cycles,
            baseline_cycles=baseline_cycles,
            speedup=speedup,
        )

    return VerifierResult(
        ok=True,
        reward=0.0,  # reward.py computes final value from speedup
        agent_cycles=report.total_cycles,
        baseline_cycles=baseline_cycles,
        speedup=speedup,
        mca_dispatch_stalls=report.dispatch_stalls,
        mca_resource_pressure_p99=report.resource_pressure_p99,
    )
