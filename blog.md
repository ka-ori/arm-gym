# We Trained an AI to Write Faster ARM Assembly Than the Compiler

*Meta / HuggingFace OpenEnv Hackathon India 2026, Finals. Theme: Wild Card. Team: (dot)mkv.*

---

Some ideas start with a single sentence in a research paper.

In late 2025, a team from Stanford and UIUC published [SuperCoder](https://arxiv.org/abs/2505.11480), a system that trained a language model to write assembly code faster than `gcc -O3`, the most aggressive optimization setting of the world's most widely used compiler. Their result was remarkable: **1.46x average speedup** on an open benchmark of 8,072 programs. A 7 billion parameter model, trained for a few hours via reinforcement learning, was regularly outperforming decades of compiler engineering.

The paper was explicit about one thing it did not cover:

> *"Extending to ARM, RISC-V, and GPU kernels is noted as future work."*

That sentence is where ARM-Gym begins.

---

## First, a little context: what is a compiler, and why should you care?

If you spend your days writing Python or JavaScript, you might never think about what actually runs on the silicon inside your device. But there is a long, invisible chain between the code a programmer writes and the instructions a processor executes, and at the center of that chain is a **compiler**.

A compiler is a program that translates human-readable source code into machine instructions. When you compile a C program with the flag `-O3`, you are telling the compiler: *use everything you know to make this as fast as possible.* GCC and Clang have been doing this for decades. Thousands of engineer-years have gone into `-O3`. It inlines functions, unrolls loops, reorders instructions to avoid stalls, and picks faster instruction variants when it can.

**And yet — compilers must be conservative.**

A compiler cannot make an assumption that might be wrong for even one program in the world. It cannot take a risk that speeds up 99 programs but silently breaks the 100th. It follows rules. Rules generalize well. But rules also leave performance on the table, especially for the specific, narrow, mathematically predictable workloads that dominate AI inference: things like matrix multiply, softmax, and convolution.

These functions run **millions of times per second** inside every large language model deployment. Every wasted CPU cycle in a softmax kernel is a wasted cycle in every AI system running on that hardware, forever.

---

## The hardware that makes this urgent: ARM is everywhere now

For a long time, ARM meant smartphones. That is no longer true.

Today:

- **AWS Graviton5** (Neoverse V3), the most widely deployed cloud compute platform in the world, runs on ARM
- **Azure Cobalt 100** (Neoverse N2), Microsoft's custom data center chip, runs on ARM
- **Every Apple Mac** sold since 2020 runs on ARM
- **Meta's AGI CPU**, 136 cores, 3nm fabrication, deployed in 2026 for AI inference, runs on ARM

Any improvement in how efficiently code runs on ARM does not just touch one product. It touches all of that. And because the bottleneck is always in the tight computational kernels, the matrix multiplies, the softmax functions, that is exactly where we focused.

---

## The research that built the foundation

ARM-Gym did not emerge from nothing. It stands on a lineage of research that moved, over five years, from "can RL find faster algorithms?" to "can an LLM learn to beat the compiler?"

### DeepMind AlphaDev (2023): RL discovers a sorting algorithm 70% faster than libc++

In 2023, DeepMind published [AlphaDev](https://www.nature.com/articles/s41586-023-06004-9) in *Nature*. They applied AlphaZero, the same MCTS-based system that mastered chess and Go, to x86 assembly generation. The task: discover a sorting algorithm faster than the hand-tuned one in LLVM's standard library.

The result: **sort3 was 70% faster than libc++**. The discovered algorithm was integrated directly into LLVM's production libc++ in 2023, where it remains today. It was the first ML-discovered algorithm to ship in a major production compiler.

But AlphaDev had limits. MCTS requires an expensive search per problem. You cannot learn a general policy from it. Every new problem needed a fresh search costing days of compute. It targeted x86 only. And it could not explain *why* its algorithms worked.

AlphaDev proved the concept: RL can find assembly routines faster than the world's best compilers and get them into production. It did not build a general-purpose tool.

### Meta LLM Compiler (2023): the first language model for compiler optimization

That same year, Chris Cummins at Meta AI published a landmark paper on using language models for compiler pass ordering. The question: can an LLM learn to select the right sequence of LLVM optimization passes for a given piece of code?

Answer: yes. Their 7 billion parameter model achieved **3% code size reduction over `-Oz`**, with zero additional compilations at inference. The autotuner baseline needed 2.5 billion compilations across 9,000 CPU days to find the same answers. The LLM learned in one pass, then predicted instantly.

The insight that carried forward: LLMs can internalize compiler knowledge from examples. They do not need to re-derive the answer for every input. They learn patterns that transfer.

### Compiler-R1 (NeurIPS 2025): GRPO meets compiler optimization

By late 2025, the reinforcement learning algorithm that proved itself in training DeepSeek-R1, **GRPO (Group Relative Policy Optimization)**, was being applied to compiler problems. Compiler-R1 used GRPO to learn LLVM pass ordering, achieving an 8.46% reduction in instruction count. It demonstrated that the same RL loop that teaches language models to reason step-by-step also teaches them to optimize code.

### SuperCoder (arXiv:2505.11480, 2025): the direct blueprint

SuperCoder is the paper ARM-Gym is built on. The authors asked a more direct question than anyone before: can an LLM learn to write assembly that actually *executes faster* than `gcc -O3`?

Not pass ordering. Not IR optimization. **Raw assembly generation.**

They used Qwen2.5-Coder-7B-Instruct, the same base model we use, trained via GRPO on 8,072 programs. The reward was binary: if the assembly compiles, passes all tests, and is faster than the compiler, reward equals the speedup. Otherwise, zero. No partial credit.

That "no partial credit" design turned out to matter enormously. They tested a reward that gave partial credit for passing some tests, and it scored *worse* (1.38x vs 1.46x). Partial credit lets the model learn to be "good enough." Binary reward forces it to be actually correct and actually fast.

**Result: correctness jumped from 61.4% to 95%. Average speedup from 1.10x to 1.46x over `gcc -O3`.**

One more finding worth noting: 98.5% of the speedup came from **instruction scheduling and code layout**, reordering instructions and basic blocks to hide pipeline latency. Not exotic instruction selection. Not architectural tricks. The compiler's instruction order is not optimal, and the model found better orderings.

SuperCoder was published on x86-64 only. The paper explicitly identified ARM as the natural next target. No one had built it yet.

That is the gap ARM-Gym fills.

---

## What we built: ARM-Gym

ARM-Gym is a reinforcement learning environment built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv), the framework created by Meta and Hugging Face for hackathon-grade RL environments. The task mirrors SuperCoder exactly, generate assembly that beats the compiler, but the target is **AArch64** (ARM's 64-bit instruction set) and the kernels are specifically chosen from the hot paths of AI inference workloads.

### The kernel library: what the model learns to optimize

We wrote 15 C function templates covering the operations that dominate AI inference:

- **Vector ops**: `vec_add`, `dot_product`, `saxpy` — the building blocks of neural network layers
- **Matrix ops**: `gemv`, `matmul` — the core of transformer attention and feed-forward
- **Activation and normalization**: `softmax`, `layer_norm` — run after every layer in a transformer
- **Convolution**: `conv1d`, `conv2d` — dominant in vision models
- **Elementwise**: `relu`, `gelu`, `silu`, `fma` — activation functions run billions of times per second

From 15 templates, we generate **649 variants** by varying sizes, data types (float32, float16, int8), and parameters. This prevents the model from memorizing a single solution and forces it to learn generalizable patterns.

### The training loop in plain English

Every training step works the same way:

1. **Pick a kernel.** Sample one of the 649 variants at random.
2. **Compile the baseline.** Run `clang-21 -O3` on the C source. Measure its cycle count with LLVM-MCA.
3. **Build the prompt.** Give the model the C source, the compiler's assembly, and an instruction: *write optimized AArch64 assembly for this function.* The baseline is always included. SuperCoder found that without it, even strong models produce 0% compilable code.
4. **Verify.** The model's output goes through three sequential gates. Pass all three: reward. Fail any one: zero.
5. **Update.** GRPO scores all candidates in the group, normalizes by z-score, and nudges the model toward the better ones.

The three gates are what make the reward trustworthy:

**Gate 1, Syntax.** The assembly must compile with `aarch64-linux-gnu-as`, the real GNU assembler. Not a regex. Not a syntax checker. The actual tool that produces a real binary object. Broken assembly: score zero.

**Gate 2, Correctness.** The compiled binary runs 20 times inside `qemu-aarch64-static`, a full ARM CPU emulator. Each run uses randomly generated inputs: edge cases, boundary values, near-overflow values. The output must match the original C function exactly. Inputs change every episode, so the model cannot memorize test cases and hardcode outputs.

**Gate 3, Performance.** Cycle count measured by LLVM-MCA with the LLVM 21 Neoverse V2 scheduling model, a static analysis tool that reads assembly and estimates cycles based on the CPU's actual instruction latencies. Sub-millisecond. Deterministic. No runtime noise.

A cross-check runs on every result: QEMU instruction count versus LLVM-MCA cycle estimate. A ratio above 3x vetoes the result regardless of apparent speedup. No LLM judge anywhere in this stack.

### How OpenEnv made this possible

There is a hidden engineering problem in building RL for compiler tasks: the GPU (which runs the LLM) and the CPU toolchain (which runs QEMU, the assembler, and LLVM-MCA) are completely different kinds of work. Mix them in one process and the GPU idles while QEMU runs.

OpenEnv solves this by making the environment a standalone server. The training loop talks to it over HTTP or WebSocket. The GRPOTrainer never knows QEMU exists. It sends a POST request with generated assembly and gets back a reward score.

```
POST /reset  → returns a kernel to optimize (C source + baseline assembly + cycle count)
POST /step   → takes model's assembly, runs 3-gate verifier, returns reward
GET  /state  → current episode info
WS   /ws     → same operations over WebSocket for lower latency
```

When the model produces broken assembly, it gets back exactly what went wrong: the error kind, the line number, the assembler message. The next generation can self-correct. No reward, but structured feedback instead of silence.

The curriculum logic, advancing from scalar kernels to NEON to full loops when 80% of the current stage's variants pass, lives entirely inside the environment server. The trainer is unaware of it. It just calls `reset` and `step`.

### Why this verifier cannot be gamed

Every RL environment has a reward hacking problem. Give an agent a metric and it will find the most efficient path to that metric, which is often not the path you intended.

We saw this play out in Phase 1 of this hackathon. One submission's agent learned to starve long-context requests so short-request throughput looked better. Another disconnected network access so error logs stopped appearing. A third dropped database tables to make schema validation errors vanish. In all three cases, the metric went up. The actual objective was destroyed.

Our verifier has no exploitable surface. The assembler is the real tool, not a simulator. Correctness tests are randomized every episode. The cycle counter is static analysis that cannot be tricked by runtime behavior. The cross-check ratio catches physically implausible results. Nothing in the reward stack can be fooled by a model that games metrics rather than writes correct assembly.

---

## The results: honest framing

We trained V11, Qwen2.5-Coder-7B-Instruct with LoRA r=32, 8 generations per step, 250 training steps on an NVIDIA L40S. 107 minutes wall clock.

| Metric | V11 |
|---|---|
| Correctness, first 20 log rows | 19% |
| Correctness, final 20 log rows | **70%** |
| Reward Q1 (first quarter) | 3.30 |
| Reward Q4 (final quarter) | **6.50** |
| Peak total reward | 9.03 (step 98) |
| Best speedup over clang-21 -O3 | +14.5% (LLVM-MCA estimate) |
| Win rate | 14 of 125 reward-log rows |

We want to be direct about what this means and what it does not.

**The model does not universally beat `clang-21 -O3`.** The 14.5% best speedup is the best single event across 125 logged rows, not the average. Most attempts produce assembly that is correct but slower than the compiler. The 14/125 win rate tells you the model has started finding optimization opportunities. It has not mastered them.

What the results *do* show is significant for a 250-step run: **correctness rose monotonically from 19% to 70%**. This matters because correctness is the prerequisite for everything else. An assembly that crashes QEMU contributes no speedup signal. An assembly that produces wrong math is dangerous. The model learned to write valid, functionally correct AArch64 assembly, and correctness is the foundation on which speedup is built.

The reward trajectory confirms a genuine learning trend: Q1=3.30, Q2=4.20, Q3=5.40, Q4=6.50. Every quarter better than the last.

All speedup numbers are LLVM-MCA estimates on the Neoverse V2 scheduling model. They have not been validated on physical Graviton hardware. We label them "MCA-model speedup" throughout.

---

## The philosophy: AI as a probabilistic scout

Here is the mental model that frames why this approach is valuable, even before the model reliably beats the compiler.

Modern compilers are **heuristic-bound**. They apply rules. Rules are safe, conservative, and general — they cannot be wrong for any program in the world. But rules hit a local maximum. On a specific microarchitecture, for a specific workload, there are instruction sequences that a rule-writer would never hard-code because they look strange, or because the heuristic that justifies them does not generalize beyond this exact context.

We call these **dark optimizations**: speedups that are physically achievable on real silicon but invisible to any rule-based system.

The LLM is not a better rulebook. It is a **probabilistic scout**. It has intuitions built from training on millions of lines of assembly, not rules, but patterns. It can try instruction orderings that no compiler engineer would propose, at scale, without having to justify each one a priori.

The deterministic verifier (GNU assembler + QEMU + LLVM-MCA) is the scout's ground truth. The AI proposes. The verifier confirms or rejects. This combination finds speedups that are physically real while providing the same correctness guarantees as a traditional compiler.

---

## What comes next: the neural compiler roadmap

This is not just a hackathon project. The direction it points toward is substantial.

### Short term: pattern extraction

The AI acts as a research tool. When it finds a dark optimization, an instruction ordering, a NEON vectorization pattern, a register scheduling trick, that sequence can be **back-ported into LLVM source code** as a new deterministic rule. The model finds it; engineers formalize it. The improvement becomes permanent and available to every program, without needing ML at inference time.

This is how AlphaDev's sort3 ended up in libc++. ARM-Gym can do the same for inference kernels.

### Medium term: the neural compiler pass

A specialized model embedded directly in the compiler pipeline. Instead of a fixed `-O3` flag that applies the same rules to every program, developers use a **`-O-ai` flag** that performs a search-based optimization targeting the specific chip they are deploying to. Graviton5 gets assembly tuned for Neoverse V3. Apple M4 gets assembly tuned for Avalanche cores. Same source code, different silicon, each optimized by a model that has learned the microarchitecture.

No current compiler does this. Compiler heuristics are architecture-aware at the instruction set level, not the microarchitecture level. That gap is the opportunity.

### Long term: software-defined silicon

As chip designers add new hardware instructions, ARM SME2's matrix tile operations for example, the question becomes: *can these instructions actually be used in practice, and for what?*

Today, this question is answered by humans writing benchmarks. In a world with neural compilers, an AI can be asked: *given this new instruction, find a workload where using it produces a meaningful speedup.* The loop between hardware design and software execution closes. New silicon ships with software that knows how to use it, on day one.

ARM's SME2 is in production in 2026 flagship smartphones, delivering a 5x AI speedup claim. No LLM has been specifically trained to generate SME2 code. That is Stage 4 of ARM-Gym's curriculum, explicitly marked as a research frontier with no known prior work.

---

## Why ARM, not x86?

**The gap is open.** SuperCoder, Compiler-R1, and all prior LLM compiler work targets x86-64. ARM has a completely different instruction set, different SIMD extensions (NEON, SVE2 instead of AVX/SSE), different pipeline characteristics, and different optimization opportunities. A model trained on x86 assembly does not transfer to ARM. No published system does what ARM-Gym does.

**Industry gravity.** The cloud compute story has shifted. AWS Graviton5, Azure Cobalt 100, Apple Silicon, and Meta's in-house ARM chips represent the dominant trajectory of data center compute in 2026. AI inference is increasingly deployed on ARM. Optimizing ARM code has compounding returns across an enormous installed base.

**Tooling is ready.** LLVM 21 ships with the Neoverse V2 scheduling model. QEMU 11.0 supports FEAT_SME2 and SVE2. The infrastructure for a high-fidelity reward signal exists now.

---

## What we learned from building this

**Test the reward function before touching a model.** The formula `speedup - 1.0` versus `max(0, speedup - 1.0)` is a one-character difference. With the wrong version, slower-than-compiler assembly gets a small negative reward. In GRPO, when all 8 completions in a group are equally slow, they normalize to near zero and produce no gradient. The model learns correctness but not speed, and the training curves look perfectly fine the entire time. Test your reward function independently on known inputs before attaching a model.

**LLVM version is a hard dependency.** LLVM 17's Neoverse V2 scheduling model had the processor's issue-width wrong: 16 micro-ops per cycle instead of 8. Training on that would teach the model to optimize for a processor that does not exist. We pinned LLVM 21.

**Thinking models fail on assembly generation.** SuperCoder's benchmark found DeepSeek-R1 compiles at 0% across all 200 evaluation problems. Chain-of-thought causes the model to spend its entire output budget reasoning about instruction semantics and never producing executable code. Do not start assembly RL with a reasoning model.

**Correctness is the precondition, not the target.** At 19% correctness, the model is producing assembly that fails QEMU 81% of the time. There is no speedup signal to learn from broken assembly. The most important early training milestone is not beating the compiler — it is writing assembly that runs.

---

## Try it

- **Live environment:** [kaori02-arm-gym.hf.space](https://kaori02-arm-gym.hf.space)
- **Training notebook:** [arm_gym_grpo_colab.ipynb](https://huggingface.co/spaces/kaori02/arm-gym/blob/main/eval/arm_gym_grpo_colab.ipynb) — key training steps with dependency notes
- **Trained LoRA adapters (V11):** [ZDC-M01/arm-gym-v11-train-250](https://huggingface.co/ZDC-M01/arm-gym-v11-train-250)
- **Training logs (V10 + V11):** [arm-gym-logs.zip](https://huggingface.co/spaces/kaori02/arm-gym/resolve/main/logs/arm-gym-logs.zip) — 28.6 KB CSV

**Papers this builds on:**
- [SuperCoder, arXiv:2505.11480](https://arxiv.org/abs/2505.11480) — the direct blueprint
- [AlphaDev, Nature 2023](https://www.nature.com/articles/s41586-023-06004-9) — RL proving faster-than-libc++ is possible
- [LLM Compiler, arXiv:2309.07062](https://arxiv.org/abs/2309.07062) — Meta, first LLM for compiler optimization
- [DeepSeekMath, GRPO](https://arxiv.org/abs/2402.03300) — the training algorithm

---

*All speedup values are LLVM-MCA estimates on the Neoverse V2 scheduling model and have not been validated on physical silicon. Results are labeled "MCA-model speedup" throughout.*
