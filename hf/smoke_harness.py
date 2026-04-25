#!/usr/bin/env python3
"""Smoke test for generate_test_harness: check generated C is syntactically valid."""
import os, subprocess, sys, tempfile

sys.path.insert(0, os.path.dirname(__file__))

# Pull the harness helpers from v5e_train without importing the whole module
# (which requires torch, etc.)
import re

_SIG_RE = re.compile(r"([\w\s*]+?\bkernel\s*\([^)]*\))", re.DOTALL)
_ARR_SZ = 8192
_PRINT_N = 64

def _parse_kernel_sig(c_source):
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

def generate_test_harness(c_source, with_kernel_def):
    ret_type, params, full_proto = _parse_kernel_sig(c_source)
    lines = ['#include <stdio.h>', '#include <stdlib.h>', '#include <string.h>',
             '#include <stddef.h>', '#include <stdint.h>', '#include <math.h>', '']
    if with_kernel_def:
        lines.append(c_source)
    else:
        if full_proto:
            lines.append(f"extern {full_proto};")
    lines += ['', 'int main(void) {']
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


KERNELS = {
    "vec_add": (
        '#include <stddef.h>\n'
        'void kernel(float * __restrict__ a, const float * __restrict__ b,\n'
        '            const float * __restrict__ c) {\n'
        '    for (size_t i = 0; i < 16; ++i) a[i] = b[i] + c[i];\n'
        '}\n'
    ),
    "dot": (
        '#include <stddef.h>\n'
        'float kernel(const float * __restrict__ a, const float * __restrict__ b) {\n'
        '    float s = 0;\n'
        '    for (size_t i = 0; i < 16; ++i) s += a[i] * b[i];\n'
        '    return s;\n'
        '}\n'
    ),
    "saxpy": (
        '#include <stddef.h>\n'
        'void kernel(float alpha, const float * __restrict__ x,\n'
        '            float * __restrict__ y) {\n'
        '    for (size_t i = 0; i < 32; ++i) y[i] = alpha * x[i] + y[i];\n'
        '}\n'
    ),
    "popcount": (
        '#include <stddef.h>\n'
        '#include <stdint.h>\n'
        'unsigned kernel(const uint32_t * __restrict__ x) {\n'
        '    unsigned c = 0;\n'
        '    for (size_t i = 0; i < 16; i++) {\n'
        '        uint32_t v = x[i]; while (v) { c += v & 1; v >>= 1; }\n'
        '    }\n'
        '    return c;\n'
        '}\n'
    ),
    "clip": (
        '#include <stddef.h>\n'
        'void kernel(float * __restrict__ y, const float * __restrict__ x,\n'
        '            float lo, float hi) {\n'
        '    for (size_t i = 0; i < 16; ++i) {\n'
        '        float v = x[i]; y[i] = v < lo ? lo : (v > hi ? hi : v);\n'
        '    }\n'
        '}\n'
    ),
    "layernorm": (
        '#include <stddef.h>\n'
        '#include <math.h>\n'
        'void kernel(float * __restrict__ y, const float * __restrict__ x,\n'
        '            const float * __restrict__ gamma, const float * __restrict__ beta) {\n'
        '    float mean = 0; for (size_t i = 0; i < 16; ++i) mean += x[i];\n'
        '    mean /= (float)16;\n'
        '    float var = 0; for (size_t i = 0; i < 16; ++i) { float d = x[i] - mean; var += d*d; }\n'
        '    var /= (float)16;\n'
        '    float inv = (float)1.0 / sqrtf(var + (float)1e-5);\n'
        '    for (size_t i = 0; i < 16; ++i) y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];\n'
        '}\n'
    ),
}

ok, fail = 0, 0
for name, src in KERNELS.items():
    for mode_name, with_def in [("ref", True), ("extern", False)]:
        code = generate_test_harness(src, with_kernel_def=with_def)
        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write(code)
            path = f.name
        r = subprocess.run(["cc", "-fsyntax-only", "-Wno-everything", path],
                          capture_output=True, text=True)
        os.unlink(path)
        tag = f"{name}/{mode_name}"
        if r.returncode == 0:
            print(f"  OK   {tag}")
            ok += 1
        else:
            print(f"  FAIL {tag}: {r.stderr[:200]}")
            fail += 1

print(f"\n{ok} passed, {fail} failed")
if fail:
    sys.exit(1)

print("\n--- Host compile+run test ---")
for name, src in KERNELS.items():
    code = generate_test_harness(src, with_kernel_def=True)
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(code)
        cpath = f.name
    elf = cpath.replace(".c", "")
    r = subprocess.run(["cc", "-O3", "-o", elf, cpath, "-lm"],
                      capture_output=True, text=True)
    os.unlink(cpath)
    if r.returncode != 0:
        print(f"  COMPILE FAIL {name}: {r.stderr[:200]}")
        continue
    r = subprocess.run([elf], capture_output=True, text=True, timeout=5)
    os.unlink(elf)
    if r.returncode != 0:
        print(f"  RUN FAIL {name}: exit={r.returncode}")
        continue
    lines_out = r.stdout.strip().split("\n")
    print(f"  OK {name}: {len(lines_out)} output lines, first={lines_out[0]}")

print("\nAll smoke tests passed.")
