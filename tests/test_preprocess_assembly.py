"""Unit tests for ``preprocess_assembly`` (no toolchain)."""

from __future__ import annotations

import pytest

from kaggle.reward_fn import SYNTAX_REWARD_FAIL, SYNTAX_REWARD_OK, preprocess_assembly, syntax_reward


def test_preprocess_appends_newline() -> None:
    s, err = preprocess_assembly("mov x0, #0")
    assert err is None
    assert s.endswith("\n")


def test_preprocess_maps_double_slash_to_at() -> None:
    s, err = preprocess_assembly("mov x0, x1 // foo\n")
    assert err is None
    assert "//" not in s
    assert "@" in s


def test_preprocess_rejects_unclosed_block_comment() -> None:
    _, err = preprocess_assembly("mov x0, x1 /* start\n")
    assert err == "unclosed_block_comment"


def test_preprocess_rejects_code_slop_import() -> None:
    _, err = preprocess_assembly("import java.util\nmov x0, x0\n")
    assert err == "lexical:high_level_code"


def test_preprocess_allows_valid_mnemonic_umaddl() -> None:
    # Do not use naive "long English word" heuristics — umaddl is 6+ letters.
    s, err = preprocess_assembly("umaddl x0, w1, w2, w3\n")
    assert err is None
    assert "umaddl" in s


def test_syntax_reward_negative_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    from kaggle import reward_fn as rf

    monkeypatch.setattr(rf, "_CACHE", {})
    monkeypatch.setattr(rf, "_DEBUG_SHOWN", 0)
    # No valid assembly, pre-reject or assemble fail
    bad = "<assembly>import numpy as np\n</assembly>"
    r = syntax_reward(completions=[[{"content": bad}]], variant_id=["t"], baseline_asm=[""])
    assert r == [SYNTAX_REWARD_FAIL]


def test_syntax_reward_ok_constant() -> None:
    assert SYNTAX_REWARD_OK == 1.0
    assert SYNTAX_REWARD_FAIL == -1.0
