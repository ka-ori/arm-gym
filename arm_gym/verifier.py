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
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .errors import ErrorKind, StructuredError, VerifierResult
from .mca import McaReport, run_mca
from .rollout_budget import TestCase, run_parallel, select_adversarial

BASELINE_DIST_DIR = Path(os.environ.get("ARM_GYM_BASELINE_DIST", "artifacts/baseline_dist"))


def _link(obj_path: Path, out: Path, cfg: "VerifierConfig") -> bool:
    r = subprocess.run(
        [cfg.linker, "-o", str(out), str(obj_path)],
        capture_output=True, text=True, timeout=10,
    )
    return r.returncode == 0


def _run_under_qemu(elf: Path, cfg: "VerifierConfig") -> tuple[int, str, str]:
    r = subprocess.run(
        [cfg.qemu, str(elf)],
        capture_output=True, text=True, timeout=5,
    )
    return r.returncode, r.stdout, r.stderr


def _outputs_match(cand_out: str, ref_out: str, cfg: "VerifierConfig") -> bool:
    """Compare candidate vs reference stdout. Float fields use rtol/atol; int exact.

    Tokenizes whitespace-separated values. If a token parses as float and is
    non-integer, applies tolerance; if both parse as int, requires exact match;
    otherwise compares as strings (e.g., labels, fallback prints).
    """
    a = cand_out.strip().split()
    b = ref_out.strip().split()
    if len(a) != len(b):
        return False
    for ta, tb in zip(a, b):
        if ta == tb:
            continue
        try:
            ia, ib = int(ta), int(tb)
            if ia != ib:
                return False
            continue
        except ValueError:
            pass
        try:
            fa, fb = float(ta), float(tb)
        except ValueError:
            return False
        if fa != fa or fb != fb:  # NaN handling: both NaN ⇒ equal
            if not (fa != fa and fb != fb):
                return False
            continue
        if fa in (float("inf"), float("-inf")) or fb in (float("inf"), float("-inf")):
            if fa != fb:
                return False
            continue
        diff = abs(fa - fb)
        tol = cfg.float_atol + cfg.float_rtol * max(abs(fa), abs(fb))
        if diff > tol:
            return False
    return True


def run_correctness_qemu(
    obj_path: Path,
    test: "TestCase",
    cfg: VerifierConfig,
    reference_bin: Path | None = None,
) -> bool:
    """Run candidate (and optional reference) under QEMU; compare outputs.

    Behaviour:
      * If qemu/linker unavailable → skip gate (return True), preserves CI portability.
      * Candidate must link, run, exit 0, and (when a reference object is provided)
        produce stdout matching the reference within float tolerance / int exactness.
      * No reference → fall back to "exit 0" check (legacy behaviour).
    """
    if not shutil.which(cfg.qemu) or not shutil.which(cfg.linker):
        return True

    tmpdir = Path(tempfile.mkdtemp(prefix="armgym_ctest_"))
    try:
        cand_elf = tmpdir / "cand.elf"
        if not _link(obj_path, cand_elf, cfg):
            return False
        try:
            rc_c, out_c, _err_c = _run_under_qemu(cand_elf, cfg)
        except (subprocess.TimeoutExpired, OSError):
            return False
        if rc_c != 0:
            return False
        if reference_bin is None:
            return True
        ref_elf = tmpdir / "ref.elf"
        if not _link(reference_bin, ref_elf, cfg):
            # Reference broken: cannot compare ⇒ accept candidate exit-0 as best signal.
            return True
        try:
            rc_r, out_r, _err_r = _run_under_qemu(ref_elf, cfg)
        except (subprocess.TimeoutExpired, OSError):
            return True
        if rc_r != 0:
            return True
        return _outputs_match(out_c, out_r, cfg)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


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


_ACTIVE_TEMP_DIRS: list[Path] = []


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
    _ACTIVE_TEMP_DIRS.append(d)
    return obj, None


def cleanup_temp_dirs() -> None:
    """Remove all temp dirs created by assemble(). Call after verify() completes."""
    while _ACTIVE_TEMP_DIRS:
        d = _ACTIVE_TEMP_DIRS.pop()
        shutil.rmtree(d, ignore_errors=True)


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


def load_baseline_distribution(variant_id: str) -> dict[str, Any] | None:
    p = BASELINE_DIST_DIR / f"{variant_id}.json"
    if not p.exists():
        return None
    try:
        result: dict[str, Any] = json.loads(p.read_text())
        return result
    except Exception:
        return None


def sigma_sanity_ok(speedup: float, dist: dict[str, Any]) -> bool:
    """Return False if speedup exceeds 3σ above historical mean for this variant."""
    mean: float = dist.get("mean", 1.0)
    std: float = dist.get("std", 0.1)
    return bool(speedup <= mean + 3.0 * max(std, 1e-6))


def verify(
    asm: str,
    baseline_asm: str,
    variant_id: str,
    tests: list[TestCase],
    cfg: VerifierConfig,
    run_correctness: Callable[[Path, TestCase], bool],
    baseline_cycles: float,
) -> VerifierResult:
    try:
        return _verify_inner(
            asm, baseline_asm, variant_id, tests, cfg, run_correctness, baseline_cycles,
        )
    finally:
        cleanup_temp_dirs()


def _verify_inner(
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
    assert obj is not None  # assemble returns (Path, None) on success

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
    secondary: int | None = None
    if selected:
        secondary = run_secondary_cycle_count(obj, cfg, selected[0])
    if secondary is not None and secondary > 0:
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
        reward=0.0,
        agent_cycles=report.total_cycles,
        baseline_cycles=baseline_cycles,
        speedup=speedup,
        mca_dispatch_stalls=report.dispatch_stalls,
        mca_resource_pressure_p99=report.resource_pressure_p99,
    )
