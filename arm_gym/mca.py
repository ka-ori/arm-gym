"""LLVM-MCA throughput + precise hazard metrics.

Cut 3 fix: `no_pipeline_hazard` defined exactly as
    mca_dispatch_stalls == 0 AND mca_resource_pressure_p99 < 1.0
Parses llvm-mca --all-stats output.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class McaReport:
    total_cycles: int
    instructions: int
    ipc: float
    dispatch_stalls: int
    resource_pressure_p99: float
    raw: str

    @property
    def no_pipeline_hazard(self) -> bool:
        # Cut 3: precise definition, not a vague bool.
        return self.dispatch_stalls == 0 and self.resource_pressure_p99 < 1.0


_CYCLES_RE = re.compile(r"Total Cycles:\s+(\d+)")
_INSN_RE = re.compile(r"Instructions:\s+(\d+)")
_IPC_RE = re.compile(r"IPC:\s+([\d.]+)")
_STALL_RE = re.compile(r"Dispatch Width Stalls?.*?:\s*(\d+)", re.IGNORECASE)


def run_mca(asm: str, mca_bin: str, mcpu: str, triple: str = "aarch64-linux-gnu") -> McaReport:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.s"
        p.write_text(asm)
        cmd = [mca_bin, f"--mtriple={triple}", f"--mcpu={mcpu}", "--all-stats",
               "--iterations=100", str(p)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except FileNotFoundError as e:
            raise RuntimeError(f"llvm-mca binary not found: {mca_bin!r}") from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"llvm-mca timed out after 30s: {e}") from e
        if r.returncode != 0:
            raise RuntimeError(
                f"llvm-mca failed (rc={r.returncode}): {r.stderr.strip()[:500]}"
            )
        return parse_mca(r.stdout)


def parse_mca(out: str) -> McaReport:
    if not out or not out.strip():
        raise RuntimeError("llvm-mca produced empty output")
    m_cyc = _CYCLES_RE.search(out)
    m_in = _INSN_RE.search(out)
    m_ipc = _IPC_RE.search(out)
    m_stall = _STALL_RE.search(out)
    if not m_cyc or not m_in:
        raise RuntimeError("could not parse llvm-mca output (missing cycles/instructions)")
    try:
        cycles = int(m_cyc.group(1))
        instructions = int(m_in.group(1))
    except ValueError as e:
        raise RuntimeError(f"non-numeric llvm-mca counters: {e}") from e
    if cycles < 0 or instructions < 0:
        raise RuntimeError(
            f"llvm-mca returned negative counters cycles={cycles} insns={instructions}"
        )
    try:
        ipc = float(m_ipc.group(1)) if m_ipc else (instructions / max(cycles, 1))
    except ValueError:
        ipc = instructions / max(cycles, 1)
    try:
        stalls = int(m_stall.group(1)) if m_stall else 0
    except ValueError:
        stalls = 0
    p99 = _extract_resource_pressure_p99(out)
    return McaReport(total_cycles=cycles, instructions=instructions, ipc=ipc,
                     dispatch_stalls=stalls, resource_pressure_p99=p99, raw=out)


def _extract_resource_pressure_p99(out: str) -> float:
    """Parse 'Resource pressure per iteration:' tables.

    Takes the 99th-percentile of per-resource pressure values.
    """
    values: list[float] = []
    in_table = False
    for line in out.splitlines():
        if "Resource pressure per iteration" in line:
            in_table = True
            continue
        if in_table:
            if not line.strip():
                break
            for tok in line.replace("|", " ").split():
                if tok in ("-",) or tok.startswith("["):
                    continue
                try:
                    values.append(float(tok))
                except ValueError:
                    continue
    if not values:
        return 0.0
    values.sort()
    idx = max(0, int(round(0.99 * (len(values) - 1))))
    return values[idx]


# NEON register regex: match v/q/d/s/h/b followed by 1-2 digits.
# v0..v31 (full 128), q0..q31 (128-bit scalar view), d0..d31 (64), s0..s31 (32),
# h0..h31 (16), b0..b31 (8). They all alias the same NEON register.
_NEON_RE = re.compile(r"\b([vqdshb])(\d{1,2})\b")
_STORE_OR_BRANCH_RE = re.compile(r"^\s*(st\w*|ret|b\w*)\b")


def uses_neon_with_liveness(asm: str) -> bool:
    """Cut 3: data-flow-aware NEON check.

    Require a NEON register to reach a live-out (consumed by a later
    instruction or stored/returned). Dead writes do not count.

    Register aliasing: v0/q0/d0/s0/h0/b0 all refer to NEON register 0.
    Normalize to the numeric suffix.
    """
    neon_defs: dict[str, int] = {}            # reg_num -> last def line
    neon_reads_after_def: set[str] = set()

    raw_lines = asm.splitlines()
    lines: list[str] = []
    for ln in raw_lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith(("//", ";", "#")):
            continue
        if s.startswith(".") and not s.startswith(".req"):
            continue  # directives
        lines.append(ln)

    for i, ln in enumerate(lines):
        parts = [p.strip() for p in ln.split(",")]
        if not parts:
            continue
        dst_regs = set(_NEON_RE.findall(parts[0]))
        src_regs: set[tuple[str, str]] = set()
        for p in parts[1:]:
            src_regs |= set(_NEON_RE.findall(p))
        is_store_or_branch = _STORE_OR_BRANCH_RE.search(ln) is not None
        if is_store_or_branch:
            # all regs on a store / branch line are sources (live-out)
            src_regs |= dst_regs
            dst_regs = set()
        for r in src_regs:
            key = r[1].lstrip("0") or "0"  # canonical number string
            if key in neon_defs:
                neon_reads_after_def.add(key)
        for r in dst_regs:
            key = r[1].lstrip("0") or "0"
            neon_defs[key] = i
    return len(neon_reads_after_def) > 0
