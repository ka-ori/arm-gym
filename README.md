# arm-gym

GRPO environment for training LLMs to emit ARM AArch64 assembly that beats `aarch64-linux-gnu-gcc -O3`.

Implements the S-tier version of the ARM-PROBLEM-STATEMENT plan — all 5 audit cuts + 6 weaker flags resolved.

## Why LLM-free verifier stack

No LLM judge anywhere in the reward loop. Verifier = assembler + LLVM-MCA + QEMU + numeric equivalence check. Reason: [`reward-hacking-lilianweng`](../meta-hackathon-llm-wiki/wiki/research/reward-hacking-lilianweng.md) — proxy reward degrades past a KL threshold, U-Sophistry. LLM-as-judge inherits sycophancy. Deterministic verifier cannot be sweet-talked.

Secondary verifier runs an independent baseline (native gcc cycle-count path) against the primary LLVM-MCA path to catch proxy drift. See [`arm_gym/verifier.py`](arm_gym/verifier.py).

## Audit fixes

| # | Cut | Location |
|---|-----|----------|
| 1 | LLVM 21 Olympus for Neoverse V3 (was clang-18) | [`Dockerfile`](Dockerfile), [`arm_gym/compile_baseline.py`](arm_gym/compile_baseline.py) |
| 2 | Procedural kernel generator 15 templates → 500-2000 variants | [`arm_gym/kernels.py`](arm_gym/kernels.py) |
| 3 | Drop hackable shaping, precise `no_pipeline_hazard` def, liveness-aware NEON | [`arm_gym/reward.py`](arm_gym/reward.py), [`arm_gym/mca.py`](arm_gym/mca.py) |
| 4 | Per-group z-score clip `[-1.5, 1.5]` + speedup clip `[1.0, 3.0]` | [`arm_gym/reward.py`](arm_gym/reward.py) |
| 5 | Smoke-test 4×L4 DDP + Unsloth + vLLM, plain-TRL fallback, no Kaggle Docker | [`scripts/smoke_4xl4.py`](scripts/smoke_4xl4.py), [`arm_gym/train.py`](arm_gym/train.py) |

Weaker flags resolved:

- Multi-agent = structured opportunity tokens + per-role reward — [`arm_gym/multi_agent.py`](arm_gym/multi_agent.py).
- 3σ bound from offline baseline distribution — [`scripts/baseline_distribution.py`](scripts/baseline_distribution.py), consumed by [`arm_gym/verifier.py`](arm_gym/verifier.py).
- Rollout budget: N=20 adversarial tests + parallel QEMU workers — [`arm_gym/rollout_budget.py`](arm_gym/rollout_budget.py).
- ARM base correctness Plan B: auto-SFT warmup if day-0 smoke <40% — [`arm_gym/train.py:maybe_sft_warmup`](arm_gym/train.py).
- `(dot)mkv` org string: placeholder, resolve pre-submission. See [`pyproject.toml`](pyproject.toml) maintainer field.
- Day-0 deliverables: scaffold is day 0. Skeleton runnable today.

Wiki free wins applied:

- `reasoning-gym` procedural pattern → [`arm_gym/kernels.py`](arm_gym/kernels.py).
- `curriculum-learning-rl` 15% episode reduction → [`arm_gym/env.py:Curriculum`](arm_gym/env.py).
- `unsloth-advanced-grpo` `lora_alpha = rank × 2`, `fast_inference=True` → [`arm_gym/train.py`](arm_gym/train.py).
- `verifier-pitfalls` 14% FN rate → stress-test note in [`tests/test_verifier_fn.py`](tests/test_verifier_fn.py).
- `openenv-turing-blog` structured error payloads → [`arm_gym/errors.py`](arm_gym/errors.py).
- `reward-hacking-lilianweng` ICRH → motivation for LLM-free verifier, stated above.

## Quick start

```bash
pip install -e .[dev]
pytest -q                              # unit regression
docker build -t arm-gym .              # requires docker + llvm-21 reachable
python scripts/smoke_4xl4.py           # validates GPU stack before training
python -m arm_gym.train --smoke        # 5-step sanity on tiny kernel
```

## Demo loop

```
  ┌─ C kernel ──► aarch64-linux-gnu-gcc -O3 ──► baseline.s + baseline_cycles
  │                                                         │
  │                                                         ▼
  └─ LLM prompt (C + baseline.s) ─► optimized.s ─► [3-gate verifier]
                                                         │
          ┌──────────────────────────────────────────────┤
          ▼ assemble fail          ▼ correctness fail    ▼ all pass
       structured                structured         agent_cycles (LLVM-MCA)
       error -> r=0               error -> r=0       speedup = baseline/agent
                                                     reward = z-clip(speedup-1)
```

## Structure

```
arm_gym/            core library (env, reward, verifier, kernels)
scripts/            smoke tests, baseline distribution builder
kernels/templates/  15 hand-written C templates (generator expands → 500-2000 variants)
tests/              pytest regression on deterministic pieces
Dockerfile          llvm-21 + qemu-user-static + aarch64 toolchain
```

## Phase-2 judging criteria mapping

- **40% Innovation** — first LLM+GRPO for ARM assembly. Confirmed greenfield (wiki `hf-arm-wiki-recipes`).
- **30% Story** — "AI beats Clang" visual demo, live HF Space.
- **20% Training evidence** — 4 plots: correctness/episode, speedup/episode, ablation (baseline-asm removed), reward-hacking incidents caught.
- **10% Pipeline** — 3-gate reward + dual verifier + LLM-free stack.
