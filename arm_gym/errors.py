"""Structured error payloads (wiki: openenv-turing-blog).

Verifier returns typed errors so the model can self-repair across generations
instead of plain-string failures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ErrorKind(str, Enum):
    ASSEMBLE_FAIL = "assemble_fail"
    LINK_FAIL = "link_fail"
    TIMEOUT = "timeout"
    SEGFAULT = "segfault"
    OUTPUT_MISMATCH = "output_mismatch"
    FLOAT_TOLERANCE = "float_tolerance"
    NAN_OR_INF = "nan_or_inf"
    MCA_FAIL = "mca_fail"
    SECONDARY_VERIFIER_MISMATCH = "secondary_verifier_mismatch"
    SPEEDUP_OUTLIER = "speedup_outlier"  # 3-sigma sanity tripped


@dataclass
class StructuredError:
    kind: ErrorKind
    message: str
    line: int | None = None
    column: int | None = None
    expected: Any = None
    actual: Any = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_prompt(self) -> str:
        """Render as JSON the model can read in the next generation."""
        import json
        return json.dumps({k: v for k, v in asdict(self).items() if v not in (None, {}, "")},
                          default=str)


@dataclass
class VerifierResult:
    ok: bool
    reward: float
    error: StructuredError | None = None
    agent_cycles: float | None = None
    baseline_cycles: float | None = None
    speedup: float | None = None
    mca_dispatch_stalls: int | None = None
    mca_resource_pressure_p99: float | None = None
