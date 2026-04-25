"""Baseline compilation for C kernels → AArch64 assembly.

Cut 1 fix: prefer LLVM 21 with -mcpu=neoverse-v2 as the Neoverse V3 proxy.
neoverse-v3 is not yet in any LLVM release (Olympus is LLVM 22, NVIDIA-specific).
Fallback is disclosed via mcpu_disclosed field, not silently swapped.

mcpu selection probes both clang and gcc to find the best CPU both compilers
actually support. Fallback chain: v3 → v2 → v1 → n2 → n1 → generic.
GCC 12 (Debian Bookworm default) supports up to neoverse-v1.
LLVM-MCA-15 supports neoverse-v1, so reward signal stays consistent.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

CLANG_CANDIDATES = ["clang-21", "clang-20", "clang-17", "clang-16", "clang-15", "clang"]
GCC_AARCH64 = "aarch64-linux-gnu-gcc"
MCA_CANDIDATES = ["llvm-mca-21", "llvm-mca-20", "llvm-mca-17", "llvm-mca-16", "llvm-mca-15", "llvm-mca"]

_MCPU_CHAIN = ["neoverse-v3", "neoverse-v2", "neoverse-v1", "neoverse-n2", "neoverse-n1", "generic"]


def find_tool(candidates: list[str]) -> str | None:
    for c in candidates:
        if shutil.which(c):
            return c
    return None


@dataclass
class ToolchainInfo:
    clang: str | None
    gcc_aarch64: str | None
    mca: str | None
    mcpu: str  # actual -mcpu used
    mcpu_disclosed: str | None  # e.g. "V1 proxy for V3" when fallback taken

    def ready(self) -> bool:
        return bool(self.clang or self.gcc_aarch64)


def detect_toolchain(preferred_cpu: str = "neoverse-v3") -> ToolchainInfo:
    clang = find_tool(CLANG_CANDIDATES)
    gcc = shutil.which(GCC_AARCH64)
    mca = find_tool(MCA_CANDIDATES)
    mcpu, disclosed = _pick_cpu(clang, gcc, preferred_cpu)
    return ToolchainInfo(clang=clang, gcc_aarch64=gcc, mca=mca, mcpu=mcpu, mcpu_disclosed=disclosed)


def _gcc_probe_mcpu(gcc: str, preferred: str) -> tuple[str, str | None]:
    """Find best mcpu the installed gcc actually accepts via test-compile."""
    chain = [preferred] + [c for c in _MCPU_CHAIN if c != preferred]
    for cpu in chain:
        try:
            r = subprocess.run(
                [gcc, f"-mcpu={cpu}", "-S", "-x", "c", "-", "-o", "/dev/null"],
                input="int f(void){return 0;}",
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                disclosed = None if cpu == preferred else f"{cpu} proxy for {preferred} (gcc limit)"
                return cpu, disclosed
        except Exception:
            continue
    return "generic", f"generic fallback for {preferred}"


def _pick_cpu(clang: str | None, gcc: str | None, preferred: str) -> tuple[str, str | None]:
    """Probe available compilers to find best mcpu both support."""
    if clang:
        try:
            # Must pass --target=aarch64-linux-gnu — without it, clang lists
            # host (x86) CPUs and neoverse-* names are not present.
            out = subprocess.run(
                [clang, "--target=aarch64-linux-gnu", "--print-supported-cpus"],
                capture_output=True, text=True, timeout=10,
            )
            supported = (out.stdout + out.stderr).lower()
            chain = [preferred] + [c for c in _MCPU_CHAIN if c != preferred]
            for cpu in chain:
                if cpu.lower() in supported:
                    disclosed = None if cpu == preferred else f"{cpu} proxy for {preferred} (clang limit)"
                    return cpu, disclosed
        except Exception:
            pass

    # No clang or clang probe failed — probe GCC directly by test-compiling.
    if gcc:
        return _gcc_probe_mcpu(gcc, preferred)

    # No compiler to probe yet; return preferred and let compile_to_asm fail loudly.
    return preferred, None


def compile_to_asm(c_source: str, tc: ToolchainInfo, opt: str = "-O3") -> str:
    """Return AArch64 assembly text for the given C source."""
    if tc.gcc_aarch64:
        return _compile_gcc(c_source, tc.gcc_aarch64, opt, tc.mcpu)
    if tc.clang:
        return _compile_clang(c_source, tc.clang, opt, tc.mcpu)
    raise RuntimeError("no AArch64 compiler available")


# HF Jobs / Kaggle often ship aarch64-linux-gnu-gcc without cutting-edge -mcpu= names
# (e.g. neoverse-v3). Try preferred from detect_toolchain, then widely-supported CPUs.
def _mcpu_try_order(preferred: str) -> list[str]:
    pool = (
        preferred,
        "neoverse-n1",
        "neoverse-v2",
        "cortex-a76",
        "cortex-a55",
        "generic",
    )
    seen: set[str] = set()
    out: list[str] = []
    for p in pool:
        pl = p.lower()
        if pl not in seen:
            seen.add(pl)
            out.append(p)
    return out


def _compile_gcc(c_src: str, gcc: str, opt: str, mcpu: str) -> str:
    last: RuntimeError | None = None
    for cpu in _mcpu_try_order(mcpu):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "k.c"
            src.write_text(c_src)
            out = Path(d) / "k.s"
            cmd = [gcc, opt, f"-mcpu={cpu}", "-S", "-fno-stack-protector",
                   "-o", str(out), str(src)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            except subprocess.TimeoutExpired as e:
                last = RuntimeError(f"gcc timed out after 30s (mcpu={cpu}): {e}")
                continue
            if r.returncode == 0:
                return out.read_text()
            last = RuntimeError(
                f"gcc failed (mcpu={cpu} rc={r.returncode}): {r.stderr.strip()[:500]}"
            )
    assert last is not None
    raise last


def _compile_clang(c_src: str, clang: str, opt: str, mcpu: str) -> str:
    last: RuntimeError | None = None
    for cpu in _mcpu_try_order(mcpu):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "k.c"
            src.write_text(c_src)
            out = Path(d) / "k.s"
            cmd = [clang, opt, "--target=aarch64-linux-gnu", f"-mcpu={cpu}",
                   "-S", "-fno-stack-protector", "-o", str(out), str(src)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            except subprocess.TimeoutExpired as e:
                last = RuntimeError(f"clang timed out after 30s (mcpu={cpu}): {e}")
                continue
            if r.returncode == 0:
                return out.read_text()
            last = RuntimeError(
                f"clang failed (mcpu={cpu} rc={r.returncode}): {r.stderr.strip()[:500]}"
            )
    assert last is not None
    raise last
