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

Every line of C code you write gets handed to a compiler. The compiler's job is to turn your human-readable logic into machine instructions that run on real hardware. It does this by applying rules, thousands of carefully validated rules, built up over decades.

The catch: those rules must be safe for every program, on every chip, in every situation. So compilers are conservative. They leave performance on the table rather than risk being wrong.

We built a reinforcement learning environment to ask a different question: *what if a model could find the sequences the compiler misses?*

We trained a 7B language model (Qwen2.5-Coder) to write AArch64 assembly that beats `clang-21 -O3`. Not by learning compiler rules, but by exploring, failing, and learning from a verifier that tells it exactly what broke and why.

In 250 steps (107 minutes, single NVIDIA L40S), correctness rose from 19% to 70%. The model started finding instruction sequences faster than the compiler.

---

## Results

| | Start | End |
|---|---|---|
| Assembly correctness (QEMU) | 19% | 70% |
| Mean reward | 3.30 | 6.50 |
| Best speedup over clang-21 -O3 | — | +14.5% (MCA estimate) |

250 steps. 107 minutes. Single L40S. 649 kernel variants.

*All speedup values are LLVM-MCA cycle estimates, not validated on physical silicon.*

---

## Materials

| | |
|---|---|
| Full writeup | [blog.md](./blog.md): research lineage, how it works, what it means |
| Training script (V11) | [hf/v11_train.py](./hf/v11_train.py): Qwen2.5-Coder-7B + LoRA r=32, GRPO, 250 steps on L40S |
| Training script (V10) | [hf/v10_train.py](./hf/v10_train.py): LoRA r=24, 200 steps, comparison run |
| Colab notebook | [eval/arm_gym_grpo_colab.ipynb](./eval/arm_gym_grpo_colab.ipynb) |
| Trained model (V11) | [ZDC-M01/arm-gym-v11-train-250](https://huggingface.co/ZDC-M01/arm-gym-v11-train-250) |
| Training logs | [arm-gym-logs.zip](https://huggingface.co/spaces/kaori02/arm-gym/resolve/main/logs/arm-gym-logs.zip) |

---

*Meta / HuggingFace OpenEnv Hackathon India 2026, Finals. Theme: Wild Card. Team: (dot)mkv.*

*All speedup values are LLVM-MCA estimates, not validated on physical silicon.*
