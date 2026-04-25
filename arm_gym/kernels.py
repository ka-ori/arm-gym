"""Kernel template registry + procedural generator.

Cut 2 fix + wiki free win 1 (reasoning-gym): 15-20 base templates expanded by
varying shapes/strides/dtypes into 500+ deterministic variants. Each variant
has a stable id (template + param hash) so held-out splits do not leak.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from hashlib import sha1
from typing import Callable, Iterable


@dataclass(frozen=True)
class KernelVariant:
    template_name: str
    params: tuple[tuple[str, int | str], ...]
    c_source: str

    @property
    def variant_id(self) -> str:
        h = sha1(f"{self.template_name}:{self.params}".encode()).hexdigest()[:12]
        return f"{self.template_name}_{h}"


@dataclass
class KernelTemplate:
    name: str
    difficulty: int  # 1=scalar, 2=neon, 3=loop, 4=sve2
    params: dict[str, list]
    render: Callable[..., str]


def _render_vec_add(n: int, dtype: str) -> str:
    return f"""\
#include <stddef.h>
void kernel({dtype} * __restrict__ a, const {dtype} * __restrict__ b,
            const {dtype} * __restrict__ c) {{
    for (size_t i = 0; i < {n}; ++i) a[i] = b[i] + c[i];
}}
"""


def _render_dot(n: int, dtype: str) -> str:
    return f"""\
#include <stddef.h>
{dtype} kernel(const {dtype} * __restrict__ a, const {dtype} * __restrict__ b) {{
    {dtype} s = 0;
    for (size_t i = 0; i < {n}; ++i) s += a[i] * b[i];
    return s;
}}
"""


def _render_matmul(m: int, n: int, k: int, dtype: str) -> str:
    return f"""\
#include <stddef.h>
void kernel({dtype} * __restrict__ c, const {dtype} * __restrict__ a,
            const {dtype} * __restrict__ b) {{
    for (size_t i = 0; i < {m}; ++i)
        for (size_t j = 0; j < {n}; ++j) {{
            {dtype} s = 0;
            for (size_t p = 0; p < {k}; ++p) s += a[i*{k}+p] * b[p*{n}+j];
            c[i*{n}+j] = s;
        }}
}}
"""


def _render_saxpy(n: int, dtype: str) -> str:
    return f"""\
#include <stddef.h>
void kernel({dtype} alpha, const {dtype} * __restrict__ x, {dtype} * __restrict__ y) {{
    for (size_t i = 0; i < {n}; ++i) y[i] = alpha * x[i] + y[i];
}}
"""


def _render_relu(n: int, dtype: str) -> str:
    return f"""\
#include <stddef.h>
void kernel({dtype} * __restrict__ y, const {dtype} * __restrict__ x) {{
    for (size_t i = 0; i < {n}; ++i) y[i] = x[i] > 0 ? x[i] : 0;
}}
"""


def _render_softmax_numer(n: int) -> str:
    return f"""\
#include <stddef.h>
#include <math.h>
void kernel(float * __restrict__ y, const float * __restrict__ x) {{
    float m = x[0];
    for (size_t i = 1; i < {n}; ++i) if (x[i] > m) m = x[i];
    for (size_t i = 0; i < {n}; ++i) y[i] = expf(x[i] - m);
}}
"""


def _render_max_reduce(n: int, dtype: str) -> str:
    return f"""\
#include <stddef.h>
{dtype} kernel(const {dtype} * __restrict__ x) {{
    {dtype} m = x[0];
    for (size_t i = 1; i < {n}; ++i) if (x[i] > m) m = x[i];
    return m;
}}
"""


def _render_l2_norm_sq(n: int, dtype: str) -> str:
    return f"""\
#include <stddef.h>
{dtype} kernel(const {dtype} * __restrict__ x) {{
    {dtype} s = 0;
    for (size_t i = 0; i < {n}; ++i) s += x[i] * x[i];
    return s;
}}
"""


def _render_transpose(m: int, n: int, dtype: str) -> str:
    return f"""\
#include <stddef.h>
void kernel({dtype} * __restrict__ b, const {dtype} * __restrict__ a) {{
    for (size_t i = 0; i < {m}; ++i)
        for (size_t j = 0; j < {n}; ++j) b[j*{m}+i] = a[i*{n}+j];
}}
"""


def _render_conv1d_3(n: int, dtype: str) -> str:
    return f"""\
#include <stddef.h>
void kernel({dtype} * __restrict__ y, const {dtype} * __restrict__ x,
            const {dtype} * __restrict__ k) {{
    for (size_t i = 0; i < {n} - 2; ++i)
        y[i] = x[i]*k[0] + x[i+1]*k[1] + x[i+2]*k[2];
}}
"""


def _render_gemv(m: int, n: int, dtype: str) -> str:
    return f"""\
#include <stddef.h>
void kernel({dtype} * __restrict__ y, const {dtype} * __restrict__ a,
            const {dtype} * __restrict__ x) {{
    for (size_t i = 0; i < {m}; ++i) {{
        {dtype} s = 0;
        for (size_t j = 0; j < {n}; ++j) s += a[i*{n}+j] * x[j];
        y[i] = s;
    }}
}}
"""


def _render_popcount(n: int) -> str:
    return f"""\
#include <stddef.h>
#include <stdint.h>
unsigned kernel(const uint32_t * __restrict__ x) {{
    unsigned c = 0;
    for (size_t i = 0; i < {n}; ++i) {{
        uint32_t v = x[i];
        while (v) {{ c += v & 1; v >>= 1; }}
    }}
    return c;
}}
"""


def _render_abs_diff(n: int, dtype: str) -> str:
    return f"""\
#include <stddef.h>
{dtype} kernel(const {dtype} * __restrict__ a, const {dtype} * __restrict__ b) {{
    {dtype} s = 0;
    for (size_t i = 0; i < {n}; ++i) {{
        {dtype} d = a[i] - b[i];
        s += d < 0 ? -d : d;
    }}
    return s;
}}
"""


def _render_clip(n: int, dtype: str) -> str:
    return f"""\
#include <stddef.h>
void kernel({dtype} * __restrict__ y, const {dtype} * __restrict__ x,
            {dtype} lo, {dtype} hi) {{
    for (size_t i = 0; i < {n}; ++i) {{
        {dtype} v = x[i];
        y[i] = v < lo ? lo : (v > hi ? hi : v);
    }}
}}
"""


def _render_elementwise_fma(n: int, dtype: str) -> str:
    return f"""\
#include <stddef.h>
void kernel({dtype} * __restrict__ y, const {dtype} * __restrict__ a,
            const {dtype} * __restrict__ b, const {dtype} * __restrict__ c) {{
    for (size_t i = 0; i < {n}; ++i) y[i] = a[i] * b[i] + c[i];
}}
"""


def _render_leaky_relu(n: int, dtype: str) -> str:
    # alpha hardcoded to a sane default (0.01) for deterministic baseline.
    return f"""\
#include <stddef.h>
void kernel({dtype} * __restrict__ y, const {dtype} * __restrict__ x, {dtype} alpha) {{
    for (size_t i = 0; i < {n}; ++i) {{
        {dtype} v = x[i];
        y[i] = v > 0 ? v : alpha * v;
    }}
}}
"""


def _render_silu(n: int, dtype: str) -> str:
    # silu(x) = x * sigmoid(x)
    expf = "expf" if dtype == "float" else "exp"
    return f"""\
#include <stddef.h>
#include <math.h>
void kernel({dtype} * __restrict__ y, const {dtype} * __restrict__ x) {{
    for (size_t i = 0; i < {n}; ++i) {{
        {dtype} v = x[i];
        y[i] = v / (({dtype})1.0 + {expf}(-v));
    }}
}}
"""


def _render_gelu_approx(n: int, dtype: str) -> str:
    # tanh-approximation: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    tanhf = "tanhf" if dtype == "float" else "tanh"
    return f"""\
#include <stddef.h>
#include <math.h>
void kernel({dtype} * __restrict__ y, const {dtype} * __restrict__ x) {{
    const {dtype} k0 = ({dtype})0.7978845608028654; /* sqrt(2/pi) */
    const {dtype} k1 = ({dtype})0.044715;
    for (size_t i = 0; i < {n}; ++i) {{
        {dtype} v = x[i];
        {dtype} u = k0 * (v + k1 * v * v * v);
        y[i] = ({dtype})0.5 * v * (({dtype})1.0 + {tanhf}(u));
    }}
}}
"""


def _render_layernorm(n: int, dtype: str) -> str:
    # gamma, beta provided as length-n arrays. Population mean/var, eps=1e-5.
    sqrtf = "sqrtf" if dtype == "float" else "sqrt"
    return f"""\
#include <stddef.h>
#include <math.h>
void kernel({dtype} * __restrict__ y, const {dtype} * __restrict__ x,
            const {dtype} * __restrict__ gamma, const {dtype} * __restrict__ beta) {{
    {dtype} mean = 0;
    for (size_t i = 0; i < {n}; ++i) mean += x[i];
    mean /= ({dtype}){n};
    {dtype} var = 0;
    for (size_t i = 0; i < {n}; ++i) {{
        {dtype} d = x[i] - mean;
        var += d * d;
    }}
    var /= ({dtype}){n};
    {dtype} inv = ({dtype})1.0 / {sqrtf}(var + ({dtype})1e-5);
    for (size_t i = 0; i < {n}; ++i) y[i] = (x[i] - mean) * inv * gamma[i] + beta[i];
}}
"""


def _render_cosine_similarity(n: int, dtype: str) -> str:
    sqrtf = "sqrtf" if dtype == "float" else "sqrt"
    return f"""\
#include <stddef.h>
#include <math.h>
{dtype} kernel(const {dtype} * __restrict__ a, const {dtype} * __restrict__ b) {{
    {dtype} dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < {n}; ++i) {{
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }}
    {dtype} denom = {sqrtf}(na) * {sqrtf}(nb);
    return denom > ({dtype})0 ? dot / denom : ({dtype})0;
}}
"""


def _render_prefix_sum(n: int, dtype: str) -> str:
    return f"""\
#include <stddef.h>
void kernel({dtype} * __restrict__ y, const {dtype} * __restrict__ x) {{
    {dtype} s = 0;
    for (size_t i = 0; i < {n}; ++i) {{
        s += x[i];
        y[i] = s;
    }}
}}
"""


# Shared knobs — widened from original to reach 500+ variants (Cut 2).
_SIZES_SMALL = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
_SIZES_MATRIX = [4, 8, 16, 32]
_SIZES_MATMUL_K = [2, 4, 8, 16, 32]
_FLOAT_DTYPES = ["float", "double"]
_INT_DTYPES = ["int", "long"]
_ALL_NUM = _FLOAT_DTYPES + _INT_DTYPES


TEMPLATES: dict[str, KernelTemplate] = {
    t.name: t for t in [
        KernelTemplate("vec_add", 1, {"n": _SIZES_SMALL, "dtype": _ALL_NUM}, _render_vec_add),
        KernelTemplate(
            "dot", 1, {"n": _SIZES_SMALL, "dtype": _FLOAT_DTYPES + _INT_DTYPES},
            _render_dot,
        ),
        KernelTemplate("saxpy", 1, {"n": _SIZES_SMALL, "dtype": _FLOAT_DTYPES}, _render_saxpy),
        KernelTemplate("relu", 1, {"n": _SIZES_SMALL, "dtype": _FLOAT_DTYPES}, _render_relu),
        KernelTemplate("max_reduce", 1, {"n": _SIZES_SMALL, "dtype": _ALL_NUM}, _render_max_reduce),
        KernelTemplate(
            "l2_norm_sq", 1, {"n": _SIZES_SMALL, "dtype": _FLOAT_DTYPES},
            _render_l2_norm_sq,
        ),
        KernelTemplate("popcount", 1, {"n": _SIZES_SMALL}, _render_popcount),
        KernelTemplate("abs_diff", 1, {"n": _SIZES_SMALL, "dtype": _ALL_NUM}, _render_abs_diff),
        KernelTemplate("clip", 1, {"n": _SIZES_SMALL, "dtype": _FLOAT_DTYPES}, _render_clip),
        KernelTemplate(
            "elementwise_fma", 2, {"n": _SIZES_SMALL, "dtype": _FLOAT_DTYPES},
            _render_elementwise_fma,
        ),
        KernelTemplate(
            "conv1d_3", 2, {"n": _SIZES_SMALL, "dtype": _FLOAT_DTYPES},
            _render_conv1d_3,
        ),
        KernelTemplate("gemv", 2, {"m": _SIZES_MATRIX, "n": _SIZES_MATRIX, "dtype": _FLOAT_DTYPES},
                       _render_gemv),
        KernelTemplate("transpose", 2, {"m": _SIZES_MATRIX, "n": _SIZES_MATRIX, "dtype": _ALL_NUM},
                       _render_transpose),
        KernelTemplate("matmul", 3, {"m": _SIZES_MATRIX, "n": _SIZES_MATRIX,
                                     "k": _SIZES_MATMUL_K, "dtype": _FLOAT_DTYPES},
                       _render_matmul),
        KernelTemplate("softmax_numer", 3, {"n": [8, 16, 32, 64, 128, 256]}, _render_softmax_numer),
        KernelTemplate(
            "leaky_relu", 1, {"n": _SIZES_SMALL, "dtype": _FLOAT_DTYPES},
            _render_leaky_relu,
        ),
        KernelTemplate(
            "silu", 2, {"n": _SIZES_SMALL, "dtype": _FLOAT_DTYPES},
            _render_silu,
        ),
        KernelTemplate(
            "gelu_approx", 2, {"n": _SIZES_SMALL, "dtype": _FLOAT_DTYPES},
            _render_gelu_approx,
        ),
        KernelTemplate(
            "layernorm", 3, {"n": _SIZES_SMALL, "dtype": _FLOAT_DTYPES},
            _render_layernorm,
        ),
        KernelTemplate(
            "cosine_similarity", 2, {"n": _SIZES_SMALL, "dtype": _FLOAT_DTYPES},
            _render_cosine_similarity,
        ),
        KernelTemplate(
            "prefix_sum", 1, {"n": _SIZES_SMALL, "dtype": _ALL_NUM},
            _render_prefix_sum,
        ),
    ]
}


def generate_variants(template_name: str) -> Iterable[KernelVariant]:
    t = TEMPLATES[template_name]
    keys = list(t.params.keys())
    for combo in itertools.product(*(t.params[k] for k in keys)):
        kwargs = dict(zip(keys, combo))
        yield KernelVariant(
            template_name=template_name,
            params=tuple(sorted(kwargs.items())),
            c_source=t.render(**kwargs),
        )


def generate_all() -> list[KernelVariant]:
    return [v for name in TEMPLATES for v in generate_variants(name)]


def split_train_eval(variants: list[KernelVariant], eval_frac: float = 0.1,
                     seed: int = 0) -> tuple[list[KernelVariant], list[KernelVariant]]:
    """Deterministic hash-based split — held-out eval IDs cannot leak."""
    rng_cut = int(eval_frac * 2**32)
    train: list[KernelVariant] = []
    evalset: list[KernelVariant] = []
    for v in variants:
        h = int(sha1(f"{seed}:{v.variant_id}".encode()).hexdigest()[:8], 16)
        (evalset if h < rng_cut else train).append(v)
    return train, evalset


def summary() -> dict:
    total = len(generate_all())
    return {"templates": len(TEMPLATES), "variants": total}
