"""No-op stopping criteria. Kept for back-compat; we rely on chat EOS instead."""
from __future__ import annotations

import torch
from transformers import StoppingCriteria


class StopOnAssemblyTag(StoppingCriteria):
    """Stub. Always returns ``False`` for every batch row."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> torch.BoolTensor:
        b = input_ids.shape[0]
        return torch.zeros(b, dtype=torch.bool, device=input_ids.device)
