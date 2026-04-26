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

**Live environment:** [kaori02-arm-gym.hf.space](https://kaori02-arm-gym.hf.space)

---

## The problem

Every line of C code you write gets handed to a compiler. The compiler translates your human-readable logic into machine instructions that actually run on hardware. It does this by applying rules, thousands of carefully validated rules, built up over decades of engineering.

The catch: those rules must be safe for every program, on every chip, in every situation. So compilers are conservative. When in doubt, they pick the safe path, not the fast one. On compute-heavy work like matrix multiplication, image processing, or neural network kernels, this conservatism leaves 20-50% of available performance untouched.

Compilers must be safe for all programs. That constraint is also what limits them.

## What we built

ARM-Gym is a reinforcement learning environment for training a language model to find the assembly sequences a compiler misses.

The model receives a C kernel, the compiler's best attempt at assembly, and a performance target. It writes its own assembly. A three-stage verifier checks it:

1. **Assemble** with the GNU assembler. Syntax errors caught here.
2. **Correctness** via QEMU, running 20 adversarial test inputs. Wrong outputs caught here.
3. **Speed** via LLVM-MCA, measuring cycle throughput on a Neoverse V2 core. Reward computed here.

The model learns by trial and error across thousands of these verifier calls. It gets rewarded when it writes assembly that is both correct and faster. It gets nothing for clever-but-wrong code. Over time, it starts to learn patterns the compiler's rule-based approach can't easily generalize.

## Training

We used GRPO (Group Relative Policy Optimization) from HuggingFace TRL, on top of Qwen2.5-Coder-7B-Instruct with a LoRA adapter.

The environment has 649 kernel variants across 15 compute templates: vector addition, dot products, softmax, matrix multiply, and more. Kernels are grouped by difficulty so the model encounters simple problems first, advancing as it gets more confident.

**V11 config:** LoRA r=32, 8 generations per step, 250 steps, single NVIDIA L40S, 107 minutes.

---

## Results

| | Start | End |
|---|---|---|
| Assembly correctness (QEMU verified) | 19% | 70% |
| Mean reward per group | 3.30 | 6.50 |
| Best speedup over clang-21 -O3 | | +14.5% (MCA estimate) |

Correctness rose monotonically across all four training quarters. The model did not plateau early. Mean reward nearly doubled. On the kernels it handled correctly, it found instruction sequences that outperformed the compiler's output on cycle estimates.

### Training curves

![Training loss over 250 steps](https://huggingface.co/spaces/kaori02/arm-gym/resolve/main/colab/results/plots/training_loss.png)

![Reward curve over 250 steps](https://huggingface.co/spaces/kaori02/arm-gym/resolve/main/colab/results/plots/reward_curve.png)

![Correctness rate over 250 steps](https://huggingface.co/spaces/kaori02/arm-gym/resolve/main/colab/results/plots/correctness_rate.png)

![Before and after: compiler vs model on a sample kernel](https://huggingface.co/spaces/kaori02/arm-gym/resolve/main/colab/results/plots/before_after_kernel.png)

*All speedup values are LLVM-MCA cycle estimates, not validated on physical silicon.*

---

## Why ARM

ARM's AArch64 instruction set powers every Apple Silicon chip, every Snapdragon phone, AWS Graviton servers, and most of the devices reading this. It is the dominant compute architecture for mobile, edge, and increasingly cloud. Compiler optimizations for AArch64, especially SIMD/NEON vectorization, are still behind x86-64 in maturity.

That gap is exactly the kind of gap an RL-trained model can exploit.

---

## Research context

This project builds directly on a line of published work:

- **AlphaDev** (DeepMind, Nature 2023): used MCTS to discover sort algorithms 70% faster than the C++ standard library. First proof that RL could beat compilers on real code.
- **Meta LLM Compiler** (Cummins et al., 2024): showed that large language models could be fine-tuned for compiler optimization tasks, achieving 3% code size reduction on LLVM passes.
- **Compiler-R1** (NeurIPS 2025): applied GRPO to LLVM pass ordering, reaching 8.46% IR reduction. First use of GRPO in the compiler optimization loop.
- **SuperCoder** (arXiv:2505.11480): applied GRPO to x86-64 assembly generation with a similar verifier setup, reaching 1.46x over gcc -O3 on select kernels.

ARM-Gym takes the SuperCoder approach and ports it to AArch64, with a custom verifier, a structured curriculum, and an open environment anyone can train against.

Read the full story in [blog.md](./blog.md).

---

## Reproduce the training

Everything is open.

| | |
|---|---|
| Training script (V11) | [hf/v11_train.py](./hf/v11_train.py): Qwen2.5-Coder-7B + LoRA r=32, GRPO, 250 steps |
| Training script (V10) | [hf/v10_train.py](./hf/v10_train.py): LoRA r=24, 200 steps, comparison baseline |
| Colab notebook | [eval/arm_gym_grpo_colab.ipynb](./eval/arm_gym_grpo_colab.ipynb): step-by-step with plots |
| Trained model (V11) | [ZDC-M01/arm-gym-v11-train-250](https://huggingface.co/ZDC-M01/arm-gym-v11-train-250) |
| Training logs | [arm-gym-logs.zip](https://huggingface.co/spaces/kaori02/arm-gym/resolve/main/logs/arm-gym-logs.zip): V10 + V11 CSV |

The live environment at [kaori02-arm-gym.hf.space](https://kaori02-arm-gym.hf.space) exposes the same OpenEnv-compatible API the training loop uses. You can call `/reset` and `/step` directly, or connect a Colab notebook to it.

---

*Meta / HuggingFace OpenEnv Hackathon India 2026, Finals. Theme: Wild Card. Team: (dot)mkv.*
