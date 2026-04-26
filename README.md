---
title: ARM-Gym
emoji: 🦾
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: GRPO env for AArch64 superoptimization
---

# ARM-Gym

**Can a language model write faster code than the world's best compiler?**

We built an environment to find out. ARM-Gym trains a 7B model via reinforcement learning to write ARM assembly that beats `clang -O3` on the operations that run inside every AI model: matrix multiply, softmax, convolution. After 250 training steps, correctness rose from 19% to 70%, and the model started winning against the compiler on specific kernels.

This is the first system of its kind for ARM. [Read the full story →](./blog.md)

---

## The problem, briefly

When you run a neural network, the compute bottleneck lives in a small set of functions: matmul, softmax, a few others. These run millions of times per second on the world's most deployed hardware, which is now ARM, not x86. AWS Graviton, Azure Cobalt, every Apple Mac since 2020, Meta's in-house AI chips — all ARM.

Compilers like Clang are remarkable engineering, but they are built to be safe for every program that has ever been written. That conservatism has a cost. On specific hardware, for specific workloads, there are instruction orderings that a human compiler engineer would never hard-code — too narrow, too risky — but that are physically faster on real silicon.

Reinforcement learning can find those orderings. [SuperCoder](https://arxiv.org/abs/2505.11480) proved this on x86 in 2025, achieving 1.46x average speedup over `gcc -O3`. Their paper noted ARM as the open gap. ARM-Gym is that gap, built.

---

## What it does

A language model receives a C function and the compiler's assembly for it. Its job: write better ARM assembly. Every attempt is put through a three-stage check before getting a score.

**Stage 1 — does it compile?** The real GNU assembler for ARM rejects anything that is not valid assembly. No regex, no simulator. If it fails here, score is zero.

**Stage 2 — does it compute the right answer?** The compiled binary runs 20 times in a hardware emulator with randomized inputs, edge cases included. The output must match the C function exactly. The test inputs change every episode, so the model cannot cheat by memorizing them.

**Stage 3 — is it actually faster?** LLVM's cycle estimator reads the assembly and measures expected throughput on an ARM Neoverse V2 processor. If the model's assembly is faster than the compiler's, the reward equals the speedup. If it is slower, the reward is zero, not negative, because GRPO learns from relative differences within a group of attempts, not from punishment.

This reward design comes directly from SuperCoder's finding that binary pass/fail consistently outperforms partial credit. The model is forced to be correct *and* fast, not just good enough.

---

## Results

250 training steps. 107 minutes. Single NVIDIA L40S. Qwen2.5-Coder-7B-Instruct + LoRA r=32.

| | Start | End |
|---|---|---|
| Assembly correctness (QEMU verified) | 19% | **70%** |
| Mean training reward (per quarter) | 3.30 | **6.50** |
| Best speedup over clang-21 -O3 | — | +14.5% (MCA estimate) |
| Steps where model beat the compiler | — | 14 of 125 logged |

The model does not universally beat the compiler. The 14.5% best speedup is the strongest single event across the run, not the average. What the numbers show is that the model *learned* — correctness rose monotonically every quarter, and with correctness comes the ability to eventually find better instruction sequences.

All speedup values are LLVM-MCA cycle estimates, not validated on physical silicon. The trained weights, logs, and training notebook are all public.

For the full breakdown, honest framing of what the numbers mean, and what comes next: [read the blog](./blog.md).

---

## Research lineage

This did not come from nowhere. Four papers made it possible.

**AlphaDev (DeepMind, Nature 2023)** used MCTS to discover a sorting algorithm 70% faster than LLVM's libc++, which was integrated into production in 2023. First proof that RL can beat a production compiler and ship the result.

**LLM Compiler (Meta, 2023)** showed that a language model could learn compiler pass ordering, achieving 3% code-size reduction with zero re-compilations at inference. First LLM applied to compiler optimization.

**Compiler-R1 (NeurIPS 2025)** applied GRPO to LLVM pass ordering, 8.46% instruction reduction. GRPO proved itself on compiler tasks.

**SuperCoder (Stanford/UIUC, 2025)** took it all the way: GRPO on raw assembly generation, 1.46x over `gcc -O3` on x86-64. Their paper said ARM was future work. ARM-Gym is that work.

---

## Where this goes

The model finds optimization opportunities that the compiler misses. In the short term, those discoveries can be back-ported into LLVM as deterministic rules, the same way AlphaDev's sort3 became a permanent part of the standard library.

In the medium term, the vision is a `-O-ai` compiler flag that performs hardware-specific search optimization at build time, giving Graviton5 deployments assembly tuned for Neoverse V3 instead of generic ARM.

The long-term direction is software-defined silicon: closing the loop so chip designers can ask an AI "if we add this instruction, can you find a workload that uses it?" before the chip is even fabricated.

ARM SME2, the matrix extension in 2026 flagship chips with a claimed 5x AI speedup, has never had an LLM specifically trained to generate code for it. That is Stage 4 in ARM-Gym's curriculum.

---

## Resources

| | |
|---|---|
| Full writeup | [blog.md](./blog.md) — research lineage, design decisions, honest results, future roadmap |
| Training notebook | [arm_gym_grpo_colab.ipynb](https://huggingface.co/spaces/kaori02/arm-gym/blob/main/eval/arm_gym_grpo_colab.ipynb) — key steps with dependency notes for Colab |
| Trained model (V11) | [ZDC-M01/arm-gym-v11-train-250](https://huggingface.co/ZDC-M01/arm-gym-v11-train-250) — LoRA r=32, 250 steps |
| Training logs | [arm-gym-logs.zip](https://huggingface.co/spaces/kaori02/arm-gym/resolve/main/logs/arm-gym-logs.zip) — V10 + V11 CSV, 28.6 KB |
| Live environment | [kaori02-arm-gym.hf.space](https://kaori02-arm-gym.hf.space) — POST /reset, POST /step, WebSocket /ws |

---

## Training

```bash
pip install -e ".[dev]"
python -m arm_gym.train --smoke   # 5-step sanity, no GPU needed
```

Full training: upload `eval/arm_gym_grpo_colab.ipynb` to Colab or Kaggle with a GPU. The notebook has a note at the top about dependency conflicts and what to expect.

---

*Meta / HuggingFace OpenEnv Hackathon India 2026, Finals. Theme: Wild Card. Team: (dot)mkv.*
*All speedup values are LLVM-MCA estimates, not validated on physical silicon.*
