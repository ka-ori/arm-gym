"""Correctness gate budget: N=20 adversarial tests + parallel QEMU workers.

Weaker flag 3 fix: 100 tests × 8 generations × QEMU ms > 500ms rollout target.
Solution: adversarial selection keeps 20 tests that historically catch the most
mutations + ThreadPoolExecutor for QEMU invocations.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

DEFAULT_N_TESTS = 20
DEFAULT_WORKERS = min(8, (os.cpu_count() or 2))

# Hard-coded adversarial scalar inputs that historically expose numerical bugs:
# signed zeros, infinities, denormals, IEEE-754 boundaries, int overflow edges.
EDGE_CASES: list[float] = [
    0.0, -0.0, 1.0, -1.0,
    float("inf"), float("-inf"),
    1e-38, 1e38,
    float(2 ** 23), float(2 ** 23 + 1),
    float(-(2 ** 31)), float(2 ** 31 - 1),
]


@dataclass
class TestCase:
    inputs: tuple
    expected: object
    adversarial_rank: float = 0.0  # higher = catches more mutations
    __test__ = False  # not a pytest collection target


def make_edge_case_tests() -> list[TestCase]:
    """Build TestCase entries from EDGE_CASES, pre-ranked above ordinary inputs.

    Callers (e.g. env.tests_for) inject these into the per-variant test pool
    so that select_adversarial's top-N pull naturally surfaces them.
    """
    return [
        TestCase(inputs=(v,), expected=None, adversarial_rank=10.0)
        for v in EDGE_CASES
    ]


def select_adversarial(all_tests: list[TestCase], n: int = DEFAULT_N_TESTS) -> list[TestCase]:
    """Pick top-N by adversarial_rank; fall back to head-N if ranks are uniform."""
    if not all_tests:
        return []
    ordered = sorted(all_tests, key=lambda t: -t.adversarial_rank)
    return ordered[:n]


def run_parallel(run_one: Callable[[TestCase], bool], tests: list[TestCase],
                 workers: int = DEFAULT_WORKERS, timeout: float = 5.0) -> tuple[bool, int]:
    """Returns (all_pass, failed_index_or_-1). Short-circuits on first failure."""
    if not tests:
        return True, -1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(run_one, t): i for i, t in enumerate(tests)}
        for fut in as_completed(futures, timeout=timeout * len(tests)):
            i = futures[fut]
            try:
                if not fut.result(timeout=timeout):
                    return False, i
            except Exception:
                return False, i
    return True, -1


def update_adversarial_ranks(tests: list[TestCase], failures_by_index: list[int]) -> None:
    """EMA of historical failure counts. Tests that keep catching bugs move up."""
    decay = 0.9
    for t in tests:
        t.adversarial_rank *= decay
    for i in failures_by_index:
        if 0 <= i < len(tests):
            tests[i].adversarial_rank += 1.0
