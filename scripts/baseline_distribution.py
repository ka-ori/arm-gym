"""Build offline baseline speedup distribution per kernel variant.

Weaker flag 2 fix: `verifier.sigma_sanity_ok` consumes `{variant_id}.json`
with {"mean": μ, "std": σ, "n": n, "quantile_99": q}.

Strategy:
 - For each kernel variant, compile at -O0, -O1, -O2, -O3 with gcc and clang.
 - Compute speedup = O0_cycles / OX_cycles for X in {1,2,3} × {gcc, clang}.
 - 6 samples per variant -> light distribution. Real runs should add fuzzed
   variants (different loop bounds) for larger n.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from arm_gym.compile_baseline import compile_to_asm, detect_toolchain
from arm_gym.kernels import generate_all
from arm_gym.mca import run_mca

OPT_LEVELS = ("-O0", "-O1", "-O2", "-O3")


def build(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    tc = detect_toolchain()
    if not tc.ready():
        print("no AArch64 compiler available", file=sys.stderr)
        return 2
    if not tc.mca:
        print("no llvm-mca available", file=sys.stderr)
        return 2
    for v in generate_all():
        samples: list[float] = []
        try:
            o0 = compile_to_asm(v.c_source, tc, opt="-O0")
            o0_cyc = run_mca(o0, tc.mca, tc.mcpu).total_cycles
        except Exception as e:
            print(f"skip {v.variant_id}: {e}", file=sys.stderr)
            continue
        for opt in OPT_LEVELS[1:]:
            try:
                asm = compile_to_asm(v.c_source, tc, opt=opt)
                cyc = run_mca(asm, tc.mca, tc.mcpu).total_cycles
                samples.append(o0_cyc / max(cyc, 1))
            except Exception:
                continue
        if len(samples) < 2:
            continue
        mean = statistics.fmean(samples)
        std = statistics.pstdev(samples) or 1e-6
        qs = sorted(samples)
        q99 = qs[min(len(qs) - 1, int(round(0.99 * (len(qs) - 1))))]
        payload = {"mean": mean, "std": std, "n": len(samples), "quantile_99": q99,
                   "samples": samples}
        (out_dir / f"{v.variant_id}.json").write_text(json.dumps(payload))
    print(f"wrote distributions to {out_dir}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("artifacts/baseline_dist"))
    args = p.parse_args()
    return build(args.out)


if __name__ == "__main__":
    sys.exit(main())
