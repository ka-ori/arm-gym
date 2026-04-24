"""Baseline compilation for C kernels → AArch64 assembly.

Cut 1 fix: prefer LLVM 21 Olympus scheduler for -mcpu=neoverse-v3. If unavailable,
fall back to -mcpu=neoverse-v2 and *disclose* (not silently downgrade).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

CLANG_CANDIDATES = ["clang-21", "clang-20", "clang"]
GCC_AARCH64 = "aarch64-linux-gnu-gcc"
MCA_CANDIDATES = ["llvm-mca-21", "llvm-mca-20", "llvm-mca"]


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
    mcpu_disclosed: str | None  # e.g. "V2 proxy for V3" when fallback taken

    def ready(self) -> bool:
        return bool(self.clang or self.gcc_aarch64)


def detect_toolchain(preferred_cpu: str = "neoverse-v3") -> ToolchainInfo:
    clang = find_tool(CLANG_CANDIDATES)
    gcc = shutil.which(GCC_AARCH64)
    mca = find_tool(MCA_CANDIDATES)
    mcpu, disclosed = _pick_cpu(clang, preferred_cpu)
    return ToolchainInfo(clang=clang, gcc_aarch64=gcc, mca=mca, mcpu=mcpu, mcpu_disclosed=disclosed)


def _pick_cpu(clang: str | None, preferred: str) -> tuple[str, str | None]:
    if not clang:
        return preferred, None
    try:
        out = subprocess.run([clang, "--print-supported-cpus"],
                             capture_output=True, text=True, timeout=10)
        supported = (out.stdout + out.stderr).lower()
    except Exception:
        return "neoverse-v2", f"V2 proxy for {preferred} (clang probe failed)"
    if preferred.lower() in supported:
        return preferred, None
    # Cut 1 disclosure: document the downgrade rather than silently proxy.
    return "neoverse-v2", f"V2 proxy for {preferred} (not in clang --print-supported-cpus)"


def compile_to_asm(c_source: str, tc: ToolchainInfo, opt: str = "-O3") -> str:
    """Return AArch64 assembly text for the given C source."""
    if tc.gcc_aarch64:
        return _compile_gcc(c_source, tc.gcc_aarch64, opt, tc.mcpu)
    if tc.clang:
        return _compile_clang(c_source, tc.clang, opt, tc.mcpu)
    raise RuntimeError("no AArch64 compiler available")


def _compile_gcc(c_src: str, gcc: str, opt: str, mcpu: str) -> str:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "k.c"
        src.write_text(c_src)
        out = Path(d) / "k.s"
        cmd = [gcc, opt, f"-mcpu={mcpu}", "-S", "-fno-stack-protector",
               "-o", str(out), str(src)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"gcc failed: {r.stderr}")
        return out.read_text()


def _compile_clang(c_src: str, clang: str, opt: str, mcpu: str) -> str:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "k.c"
        src.write_text(c_src)
        out = Path(d) / "k.s"
        cmd = [clang, opt, "--target=aarch64-linux-gnu", f"-mcpu={mcpu}",
               "-S", "-fno-stack-protector", "-o", str(out), str(src)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"clang failed: {r.stderr}")
        return out.read_text()
