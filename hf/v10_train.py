#!/usr/bin/env python3
"""ARM-Gym V10 — Qwen2.5-Coder-7B bf16+LoRA GRPO with harness-based correctness.

V10 changes over V9 (running on L40S 48GB — more headroom unlocks better config):
  - flavor: a10g-large (24GB) → l40sx1 (48GB)  (2x VRAM; eliminates OOM ceiling)
  - num_generations: 4 → 6                      (more GRPO rollout diversity;
                                                  G=6 was banned on 24GB)
  - max_completion_length: 512 → 768            (richer asm; difficulty-2 kernels
                                                  need more instructions to express
                                                  loop unrolling / NEON vectorisation)

V9 changes retained:
  - attn_implementation: "sdpa"    (memory-efficient; keep even on 48GB)
V8 changes retained:
  - temperature: 0.8               (diversity; breaks copy-baseline ceiling trap)
  - difficulty_max: 2              (ML-relevant kernels: gemv, conv1d_3, gelu_approx, etc.)

Everything else identical to V7b proven config:
  - learning_rate: 2e-6, beta: 0.01, epsilon: 0.25
  - gradient_accumulation_steps: 4, warmup_steps: 20
  - max_train: 200, G=6 (safe on L40S 48GB)

Profiles:
  ARMGYM_PROFILE=mvp  (default): 50 steps  → ZDC-M01/arm-gym-v10-mvp-50
  ARMGYM_PROFILE=long          : 200 steps → ZDC-M01/arm-gym-v10-train-200
                                  save_steps=50, save_total_limit=4

Expects: torch, transformers, trl, peft, datasets, arm_gym pre-installed.
System:  clang-21, llvm-mca-21, aarch64-linux-gnu-{as,gcc}, qemu-aarch64-static.
         PYTORCH_ALLOC_CONF=expandable_segments:True should be set in env.
"""
import csv, hashlib, json, logging, os, re, subprocess, sys, tempfile, threading, time
from dataclasses import asdict, dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("v10")
log.info("==== ARM-Gym V10 — Qwen2.5 bf16+LoRA GRPO (l40sx1, G=6, comp=768, sdpa) ====")

import torch
log.info("torch %s  CUDA %s  GPU %s",
         torch.__version__,
         torch.version.cuda if torch.cuda.is_available() else "N/A",
         torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

import transformers, trl, peft
log.info("transformers %s  trl %s  peft %s",
         transformers.__version__, trl.__version__, peft.__version__)

from arm_gym.compile_baseline import detect_toolchain, compile_to_asm
from arm_gym.kernels import TEMPLATES, generate_all, split_train_eval
tc = detect_toolchain()

# Image gcc is too old for neoverse-v3. Force clang-21 (verified working in V7).
if tc.clang and tc.gcc_aarch64:
    log.info("Disabling gcc (%s) — neoverse-v3 unsupported; using clang-21 instead",
             tc.gcc_aarch64)
    tc.gcc_aarch64 = None
log.info("ARM toolchain: clang=%s gcc=%s mca=%s mcpu=%s disclosed=%s",
         tc.clang, tc.gcc_aarch64, tc.mca, tc.mcpu, tc.mcpu_disclosed)

SMOKE_C = '#include <stddef.h>\nvoid f(int *a, int *b) { for(size_t i=0;i<4;i++) a[i]+=b[i]; }\n'
try:
    smoke_asm = compile_to_asm(SMOKE_C, tc)
    log.info("Smoke compile OK (%d bytes asm)", len(smoke_asm))
except Exception as e:
    log.error("SMOKE COMPILE FAILED: %s", e)
    log.error("This means ALL dataset rows will be empty. Aborting early.")
    sys.exit(1)


# ── CONFIG ────────────────────────────────────────────────────────────────────
# ARMGYM_PROFILE=mvp  (default): 50 steps  → hub ZDC-M01/arm-gym-v10-mvp-50
# ARMGYM_PROFILE=long          : 200 steps → hub ZDC-M01/arm-gym-v10-train-200
@dataclass
class Cfg:
    model_id: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    hub_model_id: str = "ZDC-M01/arm-gym-v10-mvp-50"
    steps: int = 50
    # G=6 safe on L40S 48GB; G=4 was the A10G 24GB limit
    num_generations: int = 6
    gradient_accumulation_steps: int = 4
    per_device_train_batch_size: int = 1
    lora_rank: int = 24
    lora_alpha: int = 48
    learning_rate: float = 2e-6
    max_prompt_length: int = 2048
    # 768 safe on L40S 48GB with sdpa; difficulty-2 kernels need room for loop unrolling
    max_completion_length: int = 768
    # v8: raised from 0.7 to break the copy-baseline ceiling trap observed in V7
    temperature: float = 0.8
    # v8: raised from 1 to include ML-relevant kernels (gemv, gelu_approx, conv1d_3, etc.)
    difficulty_max: int = 2
    # 200 variants cover 50 steps (grad_accum=4 → 4 samples/step → 200 unique before cycling)
    max_train: int = 200
    max_eval: int = 16
    warmup_steps: int = 20
    out_dir: str = "runs/v10-mvp"
    # mvp: single checkpoint at end + final lora-adapter
    save_steps: int = 50
    save_total_limit: int = 1

def _default_profile() -> str:
    if p := os.environ.get("ARMGYM_PROFILE"):
        return p.lower().strip()
    pfile = Path(__file__).resolve().with_name("PROFILE")
    if pfile.is_file():
        t = pfile.read_text().strip().lower()
        if t in ("mvp", "long"):
            return t
    return "mvp"

def _apply_profile(cfg: Cfg) -> Cfg:
    p = _default_profile()
    if p == "long":
        cfg.hub_model_id = "ZDC-M01/arm-gym-v10-train-200"
        cfg.steps = 200
        cfg.out_dir = "runs/v10-200"
        # Checkpoint every 50 steps → ~50/100/150/200; keep all 4
        cfg.save_steps = 50
        cfg.save_total_limit = 4
    return cfg

cfg = _apply_profile(Cfg())
log.info(
    "Profile: %s  hub=%s  out=%s  steps=%d  save_every=%d  keep_ckpt=%d",
    _default_profile(),
    cfg.hub_model_id,
    cfg.out_dir,
    cfg.steps,
    cfg.save_steps,
    cfg.save_total_limit,
)
log.info("Config: model=%s steps=%d G=%d temp=%.1f lr=%.2e beta=0.01 difficulty<=%d",
         cfg.model_id, cfg.steps, cfg.num_generations, cfg.temperature,
         cfg.learning_rate, cfg.difficulty_max)


# ── DATASET — SuperCoder A.3 prompt ──────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert AArch64 (aarch64-linux-gnu-gcc) assembly writer. "
    "Obey the user block exactly. Output only what is asked in the required tags."
)

def user_prompt(c_source: str, baseline_asm: str) -> str:
    return (
        "Given the following C code and assembly code, your task is to generate "
        "highly optimized AArch64 assembly code.\n\n"
        f"C Code:\n{c_source}\n\n"
        f"Assembly Code:\n{baseline_asm}\n\n"
        "Only output the optimized assembly code. Do not include any other text. "
        "Do not write any comments in the assembly code. "
        "Wrap the assembly code in <assembly></assembly> tags.\n\n"
        "Optimized Assembly Code:\n"
    )

def build_dataset(cfg, tok):
    from datasets import Dataset
    vs = [v for v in generate_all()
          if TEMPLATES[v.template_name].difficulty <= cfg.difficulty_max]
    log.info("Kernel variants (difficulty<=%d): %d", cfg.difficulty_max, len(vs))
    tv, ev = split_train_eval(vs, eval_frac=0.1, seed=0)
    tv, ev = tv[:cfg.max_train], ev[:cfg.max_eval]

    _compile_fails = [0]
    def row(v):
        try:
            basm = compile_to_asm(v.c_source, tc)
        except Exception as e:
            _compile_fails[0] += 1
            if _compile_fails[0] <= 3:
                log.warning("compile_to_asm failed [%d]: %s", _compile_fails[0], e)
            return None
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(v.c_source, basm)},
        ]
        for kw in (
            {"tokenize": False, "add_generation_prompt": True,
             "enable_thinking": False},
            {"tokenize": False, "add_generation_prompt": True},
        ):
            try:
                prompt = tok.apply_chat_template(msgs, **kw)
                break
            except TypeError:
                continue
        else:
            prompt = f"{SYSTEM_PROMPT}\n\n{msgs[1]['content']}\n"
        return {"prompt": prompt, "variant_id": v.variant_id,
                "baseline_asm": basm, "c_source": v.c_source}

    tr = [r for v in tv if (r := row(v)) is not None]
    er = [r for v in ev if (r := row(v)) is not None]
    log.info("Dataset: train=%d eval=%d", len(tr), len(er))
    if tr:
        toks = tok(tr[0]["prompt"], return_tensors="pt")
        log.info("Sample prompt tokens: %d", toks["input_ids"].shape[1])
        log.info("Prompt tail: %r", tr[0]["prompt"][-200:])
    return Dataset.from_list(tr), Dataset.from_list(er)


# ── REWARD FUNCTIONS ──────────────────────────────────────────────────────────
_ASM_RE = re.compile(r"<assembly>(.*?)</assembly>", re.DOTALL | re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CACHE: dict = {}
_BASELINE: dict = {}
_LOCK = threading.Lock()
_VCFG = None

def _clean_asm_directives(asm: str) -> str:
    """Strip directives GNU as rejects but clang emits (.addrsig, .ident)."""
    lines = []
    for line in asm.splitlines():
        stripped = line.strip()
        if stripped.startswith(".addrsig") or stripped.startswith(".ident"):
            continue
        lines.append(line)
    asm = "\n".join(lines)
    if ".arch " not in asm:
        asm = "\t.arch armv9-a+sve2+crc\n" + asm
    return asm

def _extract_asm(text):
    text = _THINK_RE.sub("", text).strip()
    m = _ASM_RE.search(text)
    if m:
        return _clean_asm_directives(m.group(1).strip())
    if "</assembly>" in text.lower():
        body = re.split(r"</assembly>", text, flags=re.IGNORECASE)[0]
        if "<assembly>" in body.lower():
            body = re.split(r"<assembly>", body, flags=re.IGNORECASE)[-1]
        return _clean_asm_directives(body.strip())
    if "<assembly>" in text.lower():
        return _clean_asm_directives(
            re.split(r"<assembly>", text, flags=re.IGNORECASE)[-1].strip())
    return _clean_asm_directives(text.strip())

def _vcfg():
    global _VCFG
    if _VCFG is None:
        from arm_gym.verifier import VerifierConfig
        _VCFG = VerifierConfig(
            mca_bin=tc.mca or "llvm-mca", assembler="aarch64-linux-gnu-as",
            linker="aarch64-linux-gnu-ld", qemu="qemu-aarch64-static",
            mcpu=tc.mcpu)
    return _VCFG


# ── CORRECTNESS HARNESS ──────────────────────────────────────────────────────
_SIG_RE = re.compile(r"([\w\s*]+?\bkernel\s*\([^)]*\))", re.DOTALL)
_HARNESS_DIR: Path | None = None
_REF_ELF_CACHE: dict[str, Path | None] = {}
_HARNESS_OBJ_CACHE: dict[str, Path | None] = {}
_CORRECTNESS_GCC = "aarch64-linux-gnu-gcc"
_CORRECTNESS_QEMU = "qemu-aarch64-static"
_ARR_SZ = 8192
_PRINT_N = 64

def _harness_dir() -> Path:
    global _HARNESS_DIR
    if _HARNESS_DIR is None:
        _HARNESS_DIR = Path(tempfile.mkdtemp(prefix="armgym_harness_"))
    return _HARNESS_DIR

def _parse_kernel_sig(c_source: str):
    m = _SIG_RE.search(c_source)
    if not m:
        return "void", [], None
    full_proto = " ".join(m.group(1).split())
    paren = full_proto.index("(")
    ret_and_name = full_proto[:paren].strip()
    ret_type = ret_and_name.rsplit("kernel", 1)[0].strip() or "void"
    raw_params_str = full_proto[paren + 1:].rstrip(")")
    params = []
    for p in raw_params_str.split(","):
        p = p.strip()
        if not p:
            continue
        is_const = "const " in p
        is_ptr = "*" in p
        clean = p.replace("const", "").replace("__restrict__", "").replace("*", "").strip()
        parts = clean.split()
        dtype = parts[0] if parts else "int"
        name = parts[-1] if len(parts) > 1 else f"p{len(params)}"
        params.append({"dtype": dtype, "name": name, "is_ptr": is_ptr,
                        "is_const": is_const, "raw": p})
    return ret_type, params, full_proto

def generate_test_harness(c_source: str, with_kernel_def: bool) -> str:
    ret_type, params, full_proto = _parse_kernel_sig(c_source)
    lines = ["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>",
             "#include <stddef.h>", "#include <stdint.h>", "#include <math.h>", ""]
    if with_kernel_def:
        lines.append(c_source)
    else:
        if full_proto:
            lines.append(f"extern {full_proto};")
    lines += ["", "int main(void) {"]
    call_args = []
    output_arrays = []
    for p in params:
        dt, nm = p["dtype"], p["name"]
        if p["is_ptr"]:
            lines.append(f"    static {dt} {nm}[{_ARR_SZ}];")
            if p["is_const"]:
                if dt in ("float", "double"):
                    lines.append(
                        f"    for (int i = 0; i < {_ARR_SZ}; i++)"
                        f" {nm}[i] = ({dt})((i % 17) + 1) * ({dt})0.25;")
                elif "uint" in dt:
                    lines.append(
                        f"    for (int i = 0; i < {_ARR_SZ}; i++)"
                        f" {nm}[i] = ({dt})((i % 31) + 1);")
                else:
                    lines.append(
                        f"    for (int i = 0; i < {_ARR_SZ}; i++)"
                        f" {nm}[i] = ({dt})((i % 17) + 1);")
            else:
                lines.append(f"    memset({nm}, 0, sizeof({nm}));")
                output_arrays.append((nm, dt))
            call_args.append(nm)
        else:
            if dt in ("float", "double"):
                lines.append(f"    {dt} {nm} = ({dt})1.5;")
            else:
                lines.append(f"    {dt} {nm} = ({dt})3;")
            call_args.append(nm)
    call_expr = f'kernel({", ".join(call_args)})'
    is_void = ret_type.strip() in ("void", "")
    if not is_void:
        lines.append(f"    {ret_type.strip()} _result = {call_expr};")
        if ret_type.strip() in ("float", "double"):
            lines.append('    printf("%.10g\\n", (double)_result);')
        elif "unsigned" in ret_type or "uint" in ret_type:
            lines.append('    printf("%u\\n", _result);')
        else:
            lines.append('    printf("%ld\\n", (long)_result);')
    else:
        lines.append(f"    {call_expr};")
        for nm, dt in output_arrays:
            if dt in ("float", "double"):
                lines.append(
                    f'    for (int i = 0; i < {_PRINT_N}; i++)'
                    f' printf("%.10g\\n", (double){nm}[i]);')
            elif "unsigned" in dt or "uint" in dt:
                lines.append(
                    f'    for (int i = 0; i < {_PRINT_N}; i++)'
                    f' printf("%u\\n", {nm}[i]);')
            else:
                lines.append(
                    f'    for (int i = 0; i < {_PRINT_N}; i++)'
                    f' printf("%ld\\n", (long){nm}[i]);')
    lines += ["    return 0;", "}", ""]
    return "\n".join(lines)

def _get_reference_elf(vid: str, c_source: str) -> Path | None:
    if vid in _REF_ELF_CACHE:
        p = _REF_ELF_CACHE[vid]
        if p and p.exists():
            return p
        if p is None:
            return None
    d = _harness_dir() / f"ref_{vid}"
    d.mkdir(parents=True, exist_ok=True)
    src = d / "combined.c"
    src.write_text(generate_test_harness(c_source, with_kernel_def=True))
    elf = d / "ref.elf"
    r = subprocess.run(
        [_CORRECTNESS_GCC, "-O3", "-static", "-o", str(elf), str(src), "-lm"],
        capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        log.warning("[harness] ref compile failed vid=%s: %s", vid, r.stderr[:300])
        _REF_ELF_CACHE[vid] = None
        return None
    _REF_ELF_CACHE[vid] = elf
    return elf

def _get_harness_obj(vid: str, c_source: str) -> Path | None:
    if vid in _HARNESS_OBJ_CACHE:
        p = _HARNESS_OBJ_CACHE[vid]
        if p and p.exists():
            return p
        if p is None:
            return None
    d = _harness_dir() / f"harn_{vid}"
    d.mkdir(parents=True, exist_ok=True)
    src = d / "harness.c"
    src.write_text(generate_test_harness(c_source, with_kernel_def=False))
    obj = d / "harness.o"
    r = subprocess.run(
        [_CORRECTNESS_GCC, "-c", "-o", str(obj), str(src)],
        capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        log.warning("[harness] harness compile failed vid=%s: %s", vid, r.stderr[:300])
        _HARNESS_OBJ_CACHE[vid] = None
        return None
    _HARNESS_OBJ_CACHE[vid] = obj
    return obj

def _link_candidate_elf(cand_obj: Path, vid: str, c_source: str) -> Path | None:
    harness_obj = _get_harness_obj(vid, c_source)
    if harness_obj is None:
        return None
    tag = hashlib.md5(str(cand_obj).encode()).hexdigest()[:8]
    d = _harness_dir() / f"cand_{vid}_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    elf = d / "cand.elf"
    r = subprocess.run(
        [_CORRECTNESS_GCC, "-static", "-o", str(elf),
         str(harness_obj), str(cand_obj), "-lm", "-lc"],
        capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        return None
    return elf

def _qemu_stdout(elf: Path) -> str | None:
    try:
        r = subprocess.run(
            [_CORRECTNESS_QEMU, str(elf)],
            capture_output=True, text=True, timeout=10)
        return r.stdout if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError):
        return None

def _outputs_match(out_a: str, out_b: str, rtol: float = 1e-4,
                   atol: float = 1e-6) -> bool:
    ta = out_a.strip().split()
    tb = out_b.strip().split()
    if len(ta) != len(tb):
        return False
    for a, b in zip(ta, tb):
        if a == b:
            continue
        try:
            fa, fb = float(a), float(b)
        except ValueError:
            return False
        if fa != fa and fb != fb:
            continue
        diff = abs(fa - fb)
        tol = atol + rtol * max(abs(fa), abs(fb))
        if diff > tol:
            return False
    return True

def _bcy(vid, basm):
    if vid not in _BASELINE:
        try:
            from arm_gym.mca import run_mca
            _BASELINE[vid] = float(run_mca(basm, _vcfg().mca_bin,
                                           _vcfg().mcpu).total_cycles)
        except Exception:
            _BASELINE[vid] = 1000.0
    return _BASELINE[vid]

@dataclass
class _E:
    assembles: bool = False
    runs: bool = False
    speedup: float = 0.0

_DBG_N = 0
_DBG_MAX = 5

def _eval(text, vid, basm, c_src=""):
    global _DBG_N
    k = hashlib.md5(f"{text}::{vid}".encode()).hexdigest()
    with _LOCK:
        if k in _CACHE:
            return _CACHE[k]
    e = _E()
    asm = _extract_asm(text)
    from arm_gym.verifier import assemble, cleanup_temp_dirs
    try:
        obj, err = assemble(asm, _vcfg())
        if err or obj is None:
            if _DBG_N < _DBG_MAX:
                with _LOCK:
                    if _DBG_N < _DBG_MAX:
                        _DBG_N += 1
                        log.info("[reward-dbg #%d] vid=%s err=%r",
                                 _DBG_N, vid,
                                 err.message[:200] if err else "None")
                        log.info("[reward-dbg] raw[:300]=%r", text[:300])
                        log.info("[reward-dbg] asm[:300]=%r", asm[:300])
            cleanup_temp_dirs()
            with _LOCK:
                _CACHE[k] = e
            return e
        e.assembles = True
        if c_src:
            ref_elf = _get_reference_elf(vid, c_src)
            cand_elf = _link_candidate_elf(obj, vid, c_src)
            if ref_elf and cand_elf:
                ref_out = _qemu_stdout(ref_elf)
                cand_out = _qemu_stdout(cand_elf)
                if ref_out is not None and cand_out is not None:
                    e.runs = _outputs_match(cand_out, ref_out)
                    if e.runs:
                        log.info("[correctness] PASS vid=%s", vid)
                    elif _DBG_N < _DBG_MAX:
                        with _LOCK:
                            if _DBG_N < _DBG_MAX:
                                _DBG_N += 1
                                log.info("[correctness] FAIL vid=%s "
                                         "ref=%r cand=%r",
                                         vid, ref_out[:200], cand_out[:200])
        if e.runs:
            from arm_gym.mca import run_mca
            bc = _bcy(vid, basm)
            rep = run_mca(asm, _vcfg().mca_bin, _vcfg().mcpu)
            e.speedup = bc / max(rep.total_cycles, 1)
        cleanup_temp_dirs()
    except Exception as ex:
        log.debug("eval err vid=%s: %s", vid, ex)
        try:
            cleanup_temp_dirs()
        except Exception:
            pass
    with _LOCK:
        _CACHE[k] = e
    return e

def _prep(completions, kw):
    texts = [c[-1]["content"] if isinstance(c, list) else str(c)
             for c in (completions or [])]
    n = len(texts)
    vids = list(kw.get("variant_id") or [""] * n)
    bs = list(kw.get("baseline_asm") or [""] * n)
    cs = list(kw.get("c_source") or [""] * n)
    if len(vids) == 1 and n > 1:
        vids *= n
        bs *= n
        cs *= n
    return texts, vids, bs, cs

_FMT_DBG_N = 0
_FMT_DBG_MAX = 8

def format_reward(prompts=None, completions=None, **kw):
    global _FMT_DBG_N
    texts, _, _, _ = _prep(completions, kw)
    if _FMT_DBG_N < _FMT_DBG_MAX and texts:
        _FMT_DBG_N += 1
        h = texts[0]
        log.info("[completion #%d] len=%d first500=%r", _FMT_DBG_N, len(h), h[:500])
        log.info("[completion #%d] last200=%r", _FMT_DBG_N, h[-200:])
    scores = []
    for text in texts:
        low = text.lower()
        s = 0.0
        if "<assembly>" in low:
            s += 0.3
        if "</assembly>" in low:
            s += 0.3
        body = _extract_asm(text)
        if len(re.sub(r"\s", "", body)) >= 20:
            s += 0.4
        if any(m in low for m in ("```", "<think>", "explain", "analysis")):
            s -= 0.5
        scores.append(max(-0.5, s))
    return scores

def syntax_reward(prompts=None, completions=None, **kw):
    t, v, b, c = _prep(completions, kw)
    return [3.0 if _eval(x, vi, bi, ci).assembles else 0.0
            for x, vi, bi, ci in zip(t, v, b, c)]

def correctness_reward(prompts=None, completions=None, **kw):
    t, v, b, c = _prep(completions, kw)
    return [5.0 if _eval(x, vi, bi, ci).runs else 0.0
            for x, vi, bi, ci in zip(t, v, b, c)]

def speedup_reward(prompts=None, completions=None, **kw):
    t, v, b, c = _prep(completions, kw)
    return [max(0.0, _eval(x, vi, bi, ci).speedup - 1.0)
            if _eval(x, vi, bi, ci).runs else 0.0
            for x, vi, bi, ci in zip(t, v, b, c)]


# ── MODEL LOADING (bf16 + LoRA — no Unsloth, no 4-bit) ───────────────────────
QWEN25_EOS_IDS = (151645, 151643)

def load_model(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )
    log.info("Model dtype: %s  device: %s",
             next(model.parameters()).dtype,
             next(model.parameters()).device)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    from peft import LoraConfig, get_peft_model
    lora = LoraConfig(
        r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    log.info("bf16+LoRA model loaded")

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.truncation_side = "left"

    eos_ids = []
    for tok_str in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.convert_tokens_to_ids(tok_str)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            if tid not in eos_ids:
                eos_ids.append(tid)
    if not eos_ids:
        eos_ids = list(QWEN25_EOS_IDS)
    tok.eos_token_id = eos_ids[0]
    model.config.eos_token_id = eos_ids[0]
    gc = getattr(model, "generation_config", None)
    if gc is not None:
        gc.eos_token_id = eos_ids[0] if len(eos_ids) == 1 else eos_ids
    log.info("EOS ids=%s  pad=%d  trunc_side=%s",
             eos_ids, tok.pad_token_id, tok.truncation_side)

    model.print_trainable_parameters()

    import types
    _base_generate = type(model).generate
    def _gc_safe_generate(self, *args, **kwargs):
        # Disable gradient checkpointing during generation (incompatible with generate())
        self.gradient_checkpointing_disable()
        try:
            return _base_generate(self, *args, **kwargs)
        finally:
            self.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
    model.generate = types.MethodType(_gc_safe_generate, model)
    log.info("Patched model.generate to toggle gradient_checkpointing off/on")

    return model, tok


# ── GRPO CONFIG ───────────────────────────────────────────────────────────────
def build_grpo_config(cfg):
    from trl import GRPOConfig
    gen_kwargs = {
        "eos_token_id": list(QWEN25_EOS_IDS),
        "top_p": 0.9,
        "top_k": 40,
    }
    p = dict(
        output_dir=cfg.out_dir,
        max_steps=cfg.steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        lr_scheduler_type="constant_with_warmup",
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        num_generations=cfg.num_generations,
        generation_batch_size=cfg.num_generations,
        max_prompt_length=cfg.max_prompt_length,
        max_completion_length=cfg.max_completion_length,
        mask_truncated_completions=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        max_grad_norm=1.0,
        temperature=cfg.temperature,
        generation_kwargs=gen_kwargs,
        loss_type="grpo",
        beta=0.01,
        epsilon=0.25,
        remove_unused_columns=False,
        logging_steps=1,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to="tensorboard",
    )
    # Drop params unsupported by the installed TRL version gracefully
    while True:
        try:
            return GRPOConfig(**p)
        except TypeError as e:
            m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
            if not m:
                raise
            log.warning("Dropping unsupported GRPOConfig param: %r", m.group(1))
            p.pop(m.group(1), None)


def _write_export_manifest(out: Path, cfg: Cfg) -> None:
    checkpoints = sorted(
        p.name for p in out.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")
    )
    manifest: dict = {
        "profile": _default_profile(),
        "export_time_epoch": int(time.time()),
        "train_config": asdict(cfg),
        "artifacts": {
            "config_json": "config.json",
            "log_csv": "log.csv",
            "lora_adapter_dir": "lora-adapter",
            "checkpoint_dirs": checkpoints,
        },
    }
    (out / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    log.info("Wrote export manifest (%d checkpoints): %s",
             len(checkpoints), out / "export_manifest.json")


# ── TRAIN ─────────────────────────────────────────────────────────────────────
out = Path(cfg.out_dir)
out.mkdir(parents=True, exist_ok=True)
(out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

model, tok = load_model(cfg)
train_ds, eval_ds = build_dataset(cfg, tok)

from trl import GRPOTrainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, syntax_reward, correctness_reward,
                  speedup_reward],
    args=build_grpo_config(cfg),
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    processing_class=tok,
)

# ── PRE-TRAINING SANITY CHECK ─────────────────────────────────────────────────
log.info("Running pre-training generation sanity check...")
with torch.no_grad():
    sample = train_ds[0]["prompt"]
    inputs = tok(sample, return_tensors="pt", truncation=True,
                 max_length=cfg.max_prompt_length).to(model.device)
    log.info("Sanity input: %d tokens, device=%s, dtype=%s",
             inputs["input_ids"].shape[1], inputs["input_ids"].device,
             next(model.parameters()).dtype)
    gen_ids = model.generate(
        **inputs, max_new_tokens=128, temperature=0.5, top_p=0.9, top_k=40,
        do_sample=True, eos_token_id=list(QWEN25_EOS_IDS),
    )
    new_ids = gen_ids[0][inputs["input_ids"].shape[1]:]
    gen_text = tok.decode(new_ids, skip_special_tokens=True)
    log.info("SANITY OUTPUT (%d tokens): %r", len(new_ids), gen_text[:500])
    if "stringodzi" in gen_text or len(set(gen_text.split())) < 5:
        log.error("BASE MODEL IS GENERATING GIBBERISH — dtype or device issue!")
    else:
        log.info("Base model generates coherent text. GRPO should learn.")

log.info("=" * 60)
log.info("TRAINING START: %d steps | G=%d | temp=%.1f | lr=%.2e | beta=0.01 | "
         "completion=%d | difficulty<=%d",
         cfg.steps, cfg.num_generations, cfg.temperature, cfg.learning_rate,
         cfg.max_completion_length, cfg.difficulty_max)
log.info("=" * 60)
t0 = time.time()
try:
    trainer.train()
finally:
    rows = getattr(trainer.state, "log_history", [])
    all_keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(out / "log.csv", "w", newline="") as f:
        if all_keys:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    log.info("Logged %d rows to log.csv", len(rows))

    elapsed = time.time() - t0
    log.info("Training done in %.0fs (%.1f min)", elapsed, elapsed / 60)

    try:
        trainer.save_model(str(out / "lora-adapter"))
        tok.save_pretrained(str(out / "lora-adapter"))
        log.info("LoRA adapter saved to %s", out / "lora-adapter")
    except Exception as e:
        log.error("Failed to save LoRA adapter: %s", e)

    _write_export_manifest(out, cfg)

    if hf_tok := os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=hf_tok)
            api.create_repo(cfg.hub_model_id, repo_type="model",
                            exist_ok=True, private=False)
            _msg = (
                f"v10 GRPO LoRA: profile={os.environ.get('ARMGYM_PROFILE', 'mvp')} "
                f"steps={cfg.steps} lr={cfg.learning_rate:.2e} "
                f"temp={cfg.temperature} G={cfg.num_generations} "
                f"comp={cfg.max_completion_length} difficulty<={cfg.difficulty_max} sdpa"
            )
            api.upload_folder(
                folder_path=str(out), repo_id=cfg.hub_model_id,
                repo_type="model", commit_message=_msg[:200],
            )
            log.info("Uploaded to https://huggingface.co/%s", cfg.hub_model_id)
        except Exception as e:
            log.error("HF Hub upload failed: %s", e)

log.info("==== V10 COMPLETE ====")
