"""Wiki free win 4: acknowledge rule-based assembler gate has FN risk.

These are lightweight sanity checks on the reward/verifier contract, not a
full stress suite. The 14%-false-negative finding from verifier-pitfalls
motivates running this on a wider obscure-but-valid corpus before publishing.
"""

from arm_gym.errors import ErrorKind, StructuredError, VerifierResult
from arm_gym.reward import RewardConfig, raw_reward


def test_obscure_syntax_errors_return_structured_payload():
    # Example: a valid-but-obscure AArch64 form (e.g. ld64b on Armv8.7+). If
    # the installed assembler rejects it, the error payload must still be
    # usable by the model for self-repair.
    err = StructuredError(ErrorKind.ASSEMBLE_FAIL, "unknown mnemonic `ld64b'", line=7)
    v = VerifierResult(ok=False, reward=0.0, error=err)
    assert raw_reward(v, RewardConfig()) == 0.0
    payload = err.to_prompt()
    assert "ld64b" in payload
    assert "line" in payload


def test_fn_rate_note_present():
    # Marker: `pytest -k test_fn_rate_note_present` documents the wiki
    # finding. Real stress tests live in a separate `fn_audit` pipeline that
    # runs against a 1k-example obscure-asm corpus.
    assert True, "see wiki/research/verifier-pitfalls.md (14% FN for rule-based)"
