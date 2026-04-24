"""arm-gym: GRPO environment for ARM AArch64 assembly superoptimization."""
__version__ = "0.1.0"

from .compile_baseline import ToolchainInfo, compile_to_asm, detect_toolchain
from .env import ARMGymEnv, CompilerAction, CompilerObservation, Curriculum, app
from .errors import ErrorKind, StructuredError, VerifierResult
from .kernels import (
    TEMPLATES,
    KernelTemplate,
    KernelVariant,
    generate_all,
    generate_variants,
    split_train_eval,
    summary,
)
from .mca import McaReport, run_mca, uses_neon_with_liveness
from .reward import RewardConfig, batch_rewards, group_zscore_clip, raw_reward
from .rollout_budget import TestCase, run_parallel, select_adversarial
from .verifier import VerifierConfig, verify

__all__ = [
    "ErrorKind", "StructuredError", "VerifierResult",
    "TEMPLATES", "KernelTemplate", "KernelVariant",
    "generate_all", "generate_variants", "split_train_eval", "summary",
    "RewardConfig", "batch_rewards", "group_zscore_clip", "raw_reward",
    "McaReport", "run_mca", "uses_neon_with_liveness",
    "TestCase", "select_adversarial", "run_parallel",
    "ToolchainInfo", "compile_to_asm", "detect_toolchain",
    "VerifierConfig", "verify",
    "ARMGymEnv", "CompilerAction", "CompilerObservation", "Curriculum", "app",
]
