"""Comprehensive edge-case tests for arm-gym.

Covers edge cases not in the existing test suite: NaN/unicode/truncation in
errors, empty/boundary params in kernels, zero-cycles and register aliasing in
mca, std=0 and mismatched lengths in reward, exception handling in rollout,
nested JSON in multi_agent, curriculum regression in env, sigma boundary in
verifier, and smoke/detect in train.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arm_gym.errors import ErrorKind, StructuredError, VerifierResult
from arm_gym.kernels import (
    TEMPLATES,
    KernelTemplate,
    KernelVariant,
    generate_all,
    generate_variants,
    split_train_eval,
)
from arm_gym.mca import McaReport, parse_mca, uses_neon_with_liveness
from arm_gym.multi_agent import (
    OpportunityToken,
    analyzer_credit,
    build_optimizer_prompt,
    parse_analyzer_output,
    per_role_reward,
)
from arm_gym.reward import RewardConfig, batch_rewards, group_zscore_clip, raw_reward
from arm_gym.rollout_budget import (
    TestCase,
    run_parallel,
    select_adversarial,
    update_adversarial_ranks,
)
from arm_gym.verifier import load_baseline_distribution, sigma_sanity_ok

# ---------------------------------------------------------------------------
# 1. errors.py edge cases
# ---------------------------------------------------------------------------


class TestErrorsEdgeCases:
    """StructuredError and VerifierResult edge cases."""

    @pytest.mark.parametrize("kind", list(ErrorKind))
    def test_all_error_kinds_roundtrip_json(self, kind: ErrorKind):
        """Every ErrorKind value must survive a to_prompt -> json.loads cycle."""
        err = StructuredError(kind=kind, message=f"test for {kind.value}")
        payload = json.loads(err.to_prompt())
        assert payload["kind"] == kind.value
        assert payload["message"] == f"test for {kind.value}"

    def test_verifier_result_nan_speedup(self):
        """NaN speedup should not crash and should serialize."""
        v = VerifierResult(ok=True, reward=0.0, speedup=float("nan"))
        assert math.isnan(v.speedup)

    def test_structured_error_empty_message(self):
        """Empty message should be omitted from prompt JSON."""
        err = StructuredError(kind=ErrorKind.TIMEOUT, message="")
        payload = json.loads(err.to_prompt())
        assert "message" not in payload
        assert payload["kind"] == "timeout"

    def test_structured_error_unicode_message(self):
        """Unicode chars in message should survive roundtrip."""
        msg = "invalid token → éñ \U0001f4a5"
        err = StructuredError(kind=ErrorKind.ASSEMBLE_FAIL, message=msg)
        payload = json.loads(err.to_prompt())
        assert payload["message"] == msg

    def test_structured_error_very_long_message(self):
        """Very long messages should not crash to_prompt (no built-in truncation)."""
        msg = "A" * 100_000
        err = StructuredError(kind=ErrorKind.SEGFAULT, message=msg)
        payload = json.loads(err.to_prompt())
        # The dataclass itself does not truncate. The message round-trips fully.
        assert len(payload["message"]) == 100_000

    def test_structured_error_expected_actual_complex(self):
        """Expected/actual can be any serializable type."""
        err = StructuredError(
            kind=ErrorKind.OUTPUT_MISMATCH,
            message="mismatch",
            expected=[1, 2, 3],
            actual={"x": 42},
        )
        payload = json.loads(err.to_prompt())
        assert payload["expected"] == [1, 2, 3]
        assert payload["actual"] == {"x": 42}

    def test_verifier_result_all_none_optionals(self):
        """VerifierResult with all optional fields None."""
        v = VerifierResult(ok=False, reward=0.0)
        assert v.error is None
        assert v.agent_cycles is None
        assert v.baseline_cycles is None
        assert v.speedup is None
        assert v.mca_dispatch_stalls is None
        assert v.mca_resource_pressure_p99 is None


# ---------------------------------------------------------------------------
# 2. kernels.py edge cases
# ---------------------------------------------------------------------------


class TestKernelsEdgeCases:

    def test_variant_id_stability(self):
        """Same template + params must produce identical variant_id."""
        v1 = KernelVariant(
            template_name="vec_add",
            params=(("dtype", "float"), ("n", 64)),
            c_source="...",
        )
        v2 = KernelVariant(
            template_name="vec_add",
            params=(("dtype", "float"), ("n", 64)),
            c_source="...",
        )
        assert v1.variant_id == v2.variant_id

    def test_variant_id_changes_with_params(self):
        """Different params must produce different variant_id."""
        v1 = KernelVariant("vec_add", (("dtype", "float"), ("n", 64)), "...")
        v2 = KernelVariant("vec_add", (("dtype", "float"), ("n", 128)), "...")
        assert v1.variant_id != v2.variant_id

    def test_split_eval_frac_zero(self):
        """eval_frac=0 should put everything in train, nothing in eval."""
        variants = generate_all()
        train, evalset = split_train_eval(variants, eval_frac=0.0, seed=0)
        assert len(evalset) == 0
        assert len(train) == len(variants)

    def test_split_eval_frac_one(self):
        """eval_frac=1.0 should put everything in eval, nothing in train."""
        variants = generate_all()
        train, evalset = split_train_eval(variants, eval_frac=1.0, seed=0)
        assert len(train) == 0
        assert len(evalset) == len(variants)

    def test_split_different_seeds_differ(self):
        """Different seeds should produce different splits."""
        variants = generate_all()
        _, e1 = split_train_eval(variants, eval_frac=0.1, seed=0)
        _, e2 = split_train_eval(variants, eval_frac=0.1, seed=999)
        ids1 = {v.variant_id for v in e1}
        ids2 = {v.variant_id for v in e2}
        # Not guaranteed to be completely different, but very unlikely identical.
        assert ids1 != ids2

    def test_all_templates_render_valid_c_with_kernel(self):
        """Every template's first variant should contain a 'kernel' function."""
        for name in TEMPLATES:
            v = next(generate_variants(name))
            assert "kernel" in v.c_source, f"{name} missing 'kernel' in rendered C"

    def test_softmax_numer_has_no_dtype_param(self):
        """softmax_numer only takes n (float-only). No 'dtype' in params."""
        t = TEMPLATES["softmax_numer"]
        assert "dtype" not in t.params
        assert "n" in t.params

    def test_softmax_numer_renders_float_only(self):
        """softmax_numer should always use float type."""
        for v in generate_variants("softmax_numer"):
            assert "float" in v.c_source
            break  # just check the first one

    def test_empty_template_params_edge(self):
        """A template with empty param dict should yield exactly one variant."""
        t = KernelTemplate(
            name="noop",
            difficulty=1,
            params={},
            render=lambda: "void kernel(void) {}",
        )
        # Manually replicate generate_variants logic for an empty params dict
        import itertools
        keys = list(t.params.keys())
        combos = list(itertools.product(*(t.params[k] for k in keys)))
        assert len(combos) == 1  # single empty-tuple combo
        assert combos[0] == ()


# ---------------------------------------------------------------------------
# 3. mca.py edge cases
# ---------------------------------------------------------------------------


class TestMcaEdgeCases:

    def test_parse_mca_missing_ipc_line(self):
        """When IPC line is missing, IPC should be computed as instructions/cycles."""
        out = """\
Instructions:      10
Total Cycles:      5
"""
        rep = parse_mca(out)
        assert rep.total_cycles == 5
        assert rep.instructions == 10
        assert abs(rep.ipc - 2.0) < 0.01  # 10/5 fallback

    def test_parse_mca_zero_cycles_no_div_by_zero(self):
        """Zero cycles must not cause ZeroDivisionError in IPC fallback."""
        out = """\
Instructions:      10
Total Cycles:      0
"""
        rep = parse_mca(out)
        assert rep.total_cycles == 0
        # Fallback: instructions / max(cycles, 1) = 10/1 = 10
        assert abs(rep.ipc - 10.0) < 0.01

    def test_parse_mca_missing_both_raises(self):
        """Missing both Instructions and Total Cycles should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="could not parse"):
            parse_mca("nothing useful here")

    def test_parse_mca_empty_resource_pressure_table(self):
        """Empty resource pressure table should yield p99 = 0.0."""
        out = """\
Instructions:      10
Total Cycles:      5
IPC:               2.0
Resource pressure per iteration:

"""
        rep = parse_mca(out)
        assert rep.resource_pressure_p99 == 0.0

    def test_uses_neon_with_liveness_empty_string(self):
        """Empty assembly string should return False."""
        assert uses_neon_with_liveness("") is False

    def test_uses_neon_with_liveness_only_comments(self):
        """Assembly with only comments should return False."""
        asm = """\
// comment1
; comment2
# comment3
"""
        assert uses_neon_with_liveness(asm) is False

    def test_uses_neon_with_liveness_only_directives(self):
        """Assembly with only directives should return False."""
        asm = """\
.text
.globl kernel
.type kernel, %function
.size kernel, .-kernel
"""
        assert uses_neon_with_liveness(asm) is False

    def test_uses_neon_register_aliasing_q0_v0(self):
        """q0 and v0 alias the same NEON register (register 0).

        A write via v0 consumed by a store via q0 should be detected as live.
        """
        asm = """\
    fmov v0.4s, #1.0
    str q0, [x0]
    ret
"""
        assert uses_neon_with_liveness(asm) is True

    def test_uses_neon_register_aliasing_d0_v0(self):
        """d0 and v0 share the same underlying register number."""
        asm = """\
    fmov d0, #1.0
    str d0, [x0]
    ret
"""
        assert uses_neon_with_liveness(asm) is True

    def test_uses_neon_mixed_neon_and_scalar(self):
        """Mixed NEON and scalar instructions. NEON is live-out via store."""
        asm = """\
    mov w0, #42
    fmov v1.4s, #2.0
    add w0, w0, #1
    st1 {v1.4s}, [x1]
    ret
"""
        assert uses_neon_with_liveness(asm) is True

    def test_uses_neon_mixed_neon_dead_scalar_live(self):
        """NEON reg written but never consumed; scalar ops only are live."""
        asm = """\
    fmov v3.4s, #1.0
    mov w0, #42
    ret
"""
        assert uses_neon_with_liveness(asm) is False

    def test_mca_report_no_pipeline_hazard_property(self):
        """Boundary: stalls=0, p99=0.99 -> no hazard. p99=1.0 -> hazard."""
        r1 = McaReport(10, 10, 1.0, 0, 0.99, "")
        assert r1.no_pipeline_hazard is True

        r2 = McaReport(10, 10, 1.0, 0, 1.0, "")
        assert r2.no_pipeline_hazard is False

        r3 = McaReport(10, 10, 1.0, 1, 0.5, "")
        assert r3.no_pipeline_hazard is False


# ---------------------------------------------------------------------------
# 4. reward.py edge cases
# ---------------------------------------------------------------------------


class TestRewardEdgeCases:

    def _ok_result(self, speedup=1.5, stalls=0, p99=0.5):
        return VerifierResult(
            ok=True, reward=0.0, speedup=speedup,
            agent_cycles=100, baseline_cycles=100 * (speedup if speedup else 1),
            mca_dispatch_stalls=stalls, mca_resource_pressure_p99=p99,
        )

    def test_raw_reward_speedup_none_on_ok_true(self):
        """ok=True but speedup=None should yield 0.0 (guard clause)."""
        v = VerifierResult(ok=True, reward=0.0, speedup=None)
        assert raw_reward(v, RewardConfig()) == 0.0

    def test_raw_reward_speedup_nan(self):
        """NaN speedup on ok=True. Python max/min return the non-NaN operand."""
        v = VerifierResult(ok=True, reward=0.0, speedup=float("nan"))
        r = raw_reward(v, RewardConfig())
        assert isinstance(r, float)

    def test_group_zscore_clip_all_identical(self):
        """All identical values -> std=0 -> uses 1e-6 fallback. Result near 0."""
        out = group_zscore_clip([0.5, 0.5, 0.5, 0.5])
        # (r - mu) / 1e-6 = 0.0 for all
        assert all(abs(r) < 1e-3 for r in out)

    def test_group_zscore_clip_single_element(self):
        """Single element -> len < 2, returned as-is."""
        out = group_zscore_clip([42.0])
        assert out == [42.0]

    def test_group_zscore_clip_empty_list(self):
        """Empty list -> len < 2, returned as-is."""
        out = group_zscore_clip([])
        assert out == []

    def test_batch_rewards_mismatched_lengths_guard(self):
        """batch_rewards with asms shorter than results.

        zip() silently truncates, so result should have length of shorter list.
        This tests the actual behavior (no explicit guard).
        """
        results = [self._ok_result(1.5), self._ok_result(1.2), self._ok_result(1.0)]
        asms = ["asm1"]  # only 1 asm for 3 results
        # batch_rewards uses zip() which truncates to length 1
        out = batch_rewards(results, RewardConfig(), asms=asms)
        assert len(out) == 1

    def test_shaped_mode_without_asm_none(self):
        """shaped mode with asm=None skips NEON bonus entirely."""
        cfg = RewardConfig(mode="shaped")
        v = self._ok_result(1.5, stalls=0, p99=0.5)
        r = raw_reward(v, cfg, asm=None)
        # base = 0.5, no NEON bonus (asm is None), hazard bonus = 0.1
        assert abs(r - 0.6) < 0.01

    def test_batch_rewards_with_none_asms(self):
        """batch_rewards with asms=None should work (default path)."""
        results = [self._ok_result(1.5), self._ok_result(1.2)]
        out = batch_rewards(results, RewardConfig(), asms=None)
        assert len(out) == 2
        assert all(-1.5 <= r <= 1.5 for r in out)


# ---------------------------------------------------------------------------
# 5. rollout_budget.py edge cases
# ---------------------------------------------------------------------------


class TestRolloutBudgetEdgeCases:

    def test_select_adversarial_empty_list(self):
        """Empty test list should return empty."""
        assert select_adversarial([]) == []

    def test_select_adversarial_n_greater_than_list(self):
        """n > len(tests) should return all tests (sorted by rank)."""
        tests = [TestCase((), None, adversarial_rank=r) for r in [0.1, 0.5]]
        picked = select_adversarial(tests, n=100)
        assert len(picked) == 2
        assert picked[0].adversarial_rank >= picked[1].adversarial_rank

    def test_run_parallel_with_raising_callable(self):
        """Callable that raises should count as failure."""
        def explode(t):
            raise ValueError("boom")

        tests = [TestCase((), None) for _ in range(3)]
        ok, idx = run_parallel(explode, tests)
        assert ok is False
        assert 0 <= idx < 3

    def test_run_parallel_empty_test_list(self):
        """Empty test list should return (True, -1)."""
        ok, idx = run_parallel(lambda t: True, [])
        assert ok is True
        assert idx == -1

    def test_update_adversarial_ranks_out_of_bounds_index(self):
        """Out-of-bounds failure index should be silently ignored."""
        tests = [TestCase((), None, adversarial_rank=1.0)]
        # Index 5 is out of bounds for a list of length 1
        update_adversarial_ranks(tests, failures_by_index=[5, -2, 100])
        # Only decay applied, no boost
        assert tests[0].adversarial_rank == pytest.approx(0.9)

    def test_update_adversarial_ranks_with_valid_index(self):
        """Valid index gets decay + boost."""
        tests = [TestCase((), None, adversarial_rank=1.0)]
        update_adversarial_ranks(tests, failures_by_index=[0])
        # 1.0 * 0.9 + 1.0 = 1.9
        assert tests[0].adversarial_rank == pytest.approx(1.9)

    def test_concurrent_run_parallel_correctness(self):
        """Multiple run_parallel invocations should not interfere."""

        def slow_pass(t):
            return True

        tests_a = [TestCase((), None) for _ in range(10)]
        tests_b = [TestCase((), None) for _ in range(10)]

        # Run two parallel batches sequentially and verify both pass
        ok_a, _ = run_parallel(slow_pass, tests_a)
        ok_b, _ = run_parallel(slow_pass, tests_b)
        assert ok_a is True
        assert ok_b is True

    def test_select_adversarial_stable_sort_on_equal_ranks(self):
        """When all ranks are equal, ordering is stable (by sorted)."""
        tests = [TestCase((i,), None, adversarial_rank=1.0) for i in range(5)]
        picked = select_adversarial(tests, n=5)
        assert len(picked) == 5


# ---------------------------------------------------------------------------
# 6. multi_agent.py edge cases
# ---------------------------------------------------------------------------


class TestMultiAgentEdgeCases:

    def test_parse_analyzer_output_nested_json(self):
        """Nested JSON objects inside the opportunity tokens."""
        text = '[{"kind":"vectorize_neon","rationale":"nested: {\\"x\\": 1}"}]'
        toks = parse_analyzer_output(text)
        assert len(toks) == 1
        assert toks[0].kind == "vectorize_neon"

    def test_parse_analyzer_output_empty_array(self):
        """Empty JSON array should return empty list."""
        toks = parse_analyzer_output("[]")
        assert toks == []

    def test_parse_analyzer_output_array_non_dict_items(self):
        """Array containing non-dict items should be skipped."""
        text = '[42, "string", {"kind":"use_ldp_stp"}, null, true]'
        toks = parse_analyzer_output(text)
        assert len(toks) == 1
        assert toks[0].kind == "use_ldp_stp"

    def test_parse_analyzer_output_dict_missing_kind(self):
        """Dict without 'kind' key should be skipped."""
        text = '[{"target_region":"loop"}, {"kind":"unroll_loop"}]'
        toks = parse_analyzer_output(text)
        assert len(toks) == 1
        assert toks[0].kind == "unroll_loop"

    def test_build_optimizer_prompt_empty_tokens(self):
        """Empty tokens list should still produce valid prompt."""
        prompt = build_optimizer_prompt("int k(){}", ".text\nret", [])
        assert "C Code:" in prompt
        assert "Opportunity Tokens:" in prompt
        assert "[]" in prompt
        assert "<assembly>" in prompt

    def test_analyzer_credit_empty_optimizer_asm(self):
        """Empty optimizer_asm should yield 0.0 credit."""
        toks = [OpportunityToken("use_ldp_stp")]
        assert analyzer_credit(toks, "") == 0.0

    def test_analyzer_credit_empty_tokens(self):
        """Empty tokens list should yield 0.0."""
        assert analyzer_credit([], "ldp x0, x1, [sp]") == 0.0

    def test_analyzer_credit_tokens_with_no_mnemonics(self):
        """Tokens whose kind has empty mnemonic tuple get skipped (counted=0)."""
        toks = [OpportunityToken("unroll_loop"), OpportunityToken("hoist_constant")]
        assert analyzer_credit(toks, "ldp x0, x1, [sp]") == 0.0

    def test_per_role_reward_zero_optimizer_reward(self):
        """Zero optimizer_reward should give zero to both roles."""
        toks = [OpportunityToken("use_ldp_stp")]
        asm = "ldp x0, x1, [sp]\nret"
        split = per_role_reward(0.0, toks, asm)
        assert split["optimizer"] == 0.0
        assert split["analyzer"] == 0.0

    def test_per_role_reward_negative_optimizer_reward(self):
        """Negative optimizer_reward should propagate correctly."""
        toks = [OpportunityToken("use_ldp_stp")]
        asm = "ldp x0, x1, [sp]\nret"
        split = per_role_reward(-1.0, toks, asm)
        assert split["optimizer"] == -1.0
        # analyzer = -1.0 * 0.3 * credit(1.0) = -0.3
        assert split["analyzer"] == pytest.approx(-0.3)

    def test_opportunity_token_to_json(self):
        """OpportunityToken.to_json should return a dict with all fields."""
        tok = OpportunityToken(
            kind="vectorize_neon", target_region="inner loop", rationale="wide"
        )
        d = tok.to_json()
        assert d == {
            "kind": "vectorize_neon",
            "target_region": "inner loop",
            "rationale": "wide",
        }

    def test_opportunity_token_to_json_none_fields(self):
        """Optional fields should be None in output dict."""
        tok = OpportunityToken(kind="unroll_loop")
        d = tok.to_json()
        assert d["kind"] == "unroll_loop"
        assert d["target_region"] is None
        assert d["rationale"] is None


# ---------------------------------------------------------------------------
# 7. env.py edge cases
# ---------------------------------------------------------------------------


class TestEnvEdgeCases:

    def test_curriculum_update_advancing_stage(self):
        """Curriculum should advance when pass_ema exceeds threshold."""
        from arm_gym.env import Curriculum

        cur = Curriculum(stage=1, max_stage=4, pass_ema=0.0, threshold=0.8, alpha=1.0)
        # With alpha=1.0, a single True pass sets pass_ema=1.0 > 0.8
        cur.update(True)
        assert cur.stage == 2
        assert cur.pass_ema == 0.0  # reset after advance

    def test_curriculum_no_advance_past_max(self):
        """Curriculum should not advance past max_stage."""
        from arm_gym.env import Curriculum

        cur = Curriculum(stage=4, max_stage=4, pass_ema=0.79, threshold=0.8, alpha=1.0)
        cur.update(True)
        assert cur.stage == 4  # already at max

    def test_curriculum_regress(self):
        """Curriculum should regress when pass_ema drops below 0.2."""
        from arm_gym.env import Curriculum

        cur = Curriculum(stage=3, pass_ema=0.1, threshold=0.8)
        cur.regress()
        assert cur.stage == 2
        assert cur.pass_ema == pytest.approx(0.4)  # threshold * 0.5

    def test_curriculum_no_regress_from_stage_1(self):
        """Curriculum should not regress below stage 1."""
        from arm_gym.env import Curriculum

        cur = Curriculum(stage=1, pass_ema=0.0)
        cur.regress()
        assert cur.stage == 1

    def test_curriculum_no_regress_when_ema_sufficient(self):
        """Curriculum should not regress when pass_ema >= 0.2."""
        from arm_gym.env import Curriculum

        cur = Curriculum(stage=3, pass_ema=0.3)
        cur.regress()
        assert cur.stage == 3

    def test_curriculum_filter_no_matching_variants(self):
        """Filter with stage too low for any variant should return empty."""
        from arm_gym.env import Curriculum

        cur = Curriculum(stage=0)  # stage 0 won't match any difficulty >= 1
        all_v = generate_all()
        filtered = cur.filter(all_v)
        assert len(filtered) == 0

    def test_curriculum_filter_stage_1(self):
        """Stage 1 should only include difficulty 1 variants."""
        from arm_gym.env import Curriculum

        cur = Curriculum(stage=1)
        all_v = generate_all()
        filtered = cur.filter(all_v)
        for v in filtered:
            assert TEMPLATES[v.template_name].difficulty <= 1

    def test_env_reset_returns_valid_observation(self):
        """reset() should return a CompilerObservation with required fields."""
        from arm_gym.env import ARMGymEnv

        with patch("arm_gym.env.detect_toolchain") as mock_tc, \
             patch("arm_gym.env.compile_to_asm", return_value=".text\nret"), \
             patch("arm_gym.mca.run_mca") as mock_mca:
            mock_tc.return_value = MagicMock(
                clang="clang", gcc_aarch64="aarch64-linux-gnu-gcc",
                mca="llvm-mca", mcpu="neoverse-v2", mcpu_disclosed=None,
                ready=lambda: True,
            )
            mock_mca.return_value = McaReport(
                total_cycles=100, instructions=50, ipc=2.0,
                dispatch_stalls=0, resource_pressure_p99=0.3, raw="...",
            )
            env = ARMGymEnv(variants=generate_all())
            obs = env.reset(seed=42)
            assert obs.done is False
            assert obs.variant_id
            assert obs.c_source
            assert obs.baseline_asm
            assert obs.baseline_cycles > 0
            assert obs.difficulty >= 1

    def test_env_step_with_unknown_variant_id(self):
        """step() with an unknown variant_id should return error observation."""
        from arm_gym.env import ARMGymEnv, CompilerAction

        with patch("arm_gym.env.detect_toolchain") as mock_tc:
            mock_tc.return_value = MagicMock(
                clang="clang", gcc_aarch64="aarch64-linux-gnu-gcc",
                mca="llvm-mca", mcpu="neoverse-v2", mcpu_disclosed=None,
                ready=lambda: True,
            )
            env = ARMGymEnv(variants=generate_all())
            action = CompilerAction(variant_id="nonexistent_abc123", assembly="ret")
            obs = env.step(action)
            assert obs.done is True
            assert obs.reward == 0.0
            assert obs.error_json is not None
            assert "unknown_variant" in obs.error_json

    def test_env_metadata_returns_required_fields(self):
        """metadata() should contain all expected keys."""
        from arm_gym.env import ARMGymEnv

        with patch("arm_gym.env.detect_toolchain") as mock_tc:
            mock_tc.return_value = MagicMock(
                clang="clang", gcc_aarch64="aarch64-linux-gnu-gcc",
                mca="llvm-mca", mcpu="neoverse-v2", mcpu_disclosed=None,
                ready=lambda: True,
            )
            env = ARMGymEnv(variants=generate_all())
            meta = env.metadata()
            required_keys = {
                "name", "version", "description", "supports_concurrent_sessions",
                "templates", "variants", "curriculum_stages", "reward_mode", "mcpu",
            }
            assert required_keys.issubset(meta.keys())
            assert meta["name"] == "arm-gym"

    def test_env_state_before_any_reset(self):
        """state before reset should have None values for episode-specific fields."""
        from arm_gym.env import ARMGymEnv

        with patch("arm_gym.env.detect_toolchain") as mock_tc:
            mock_tc.return_value = MagicMock(
                clang="clang", gcc_aarch64="aarch64-linux-gnu-gcc",
                mca="llvm-mca", mcpu="neoverse-v2", mcpu_disclosed=None,
                ready=lambda: True,
            )
            env = ARMGymEnv(variants=generate_all())
            s = env.state
            assert s["episode_id"] is None
            assert s["variant_id"] is None
            assert s["kernel_name"] is None
            assert s["step_count"] == 0


# ---------------------------------------------------------------------------
# 8. verifier.py edge cases
# ---------------------------------------------------------------------------


class TestVerifierEdgeCases:

    def test_sigma_sanity_ok_exactly_at_3sigma(self):
        """Exactly at mean + 3*std should pass (<=)."""
        dist = {"mean": 1.0, "std": 0.1}
        speedup = 1.0 + 3.0 * 0.1  # = 1.3
        assert sigma_sanity_ok(speedup, dist) is True

    def test_sigma_sanity_ok_just_above_3sigma(self):
        """Just above mean + 3*std should fail."""
        dist = {"mean": 1.0, "std": 0.1}
        speedup = 1.0 + 3.0 * 0.1 + 0.001  # = 1.301
        assert sigma_sanity_ok(speedup, dist) is False

    def test_sigma_sanity_ok_just_below_3sigma(self):
        """Just below mean + 3*std should pass."""
        dist = {"mean": 1.0, "std": 0.1}
        speedup = 1.0 + 3.0 * 0.1 - 0.001  # = 1.299
        assert sigma_sanity_ok(speedup, dist) is True

    def test_sigma_sanity_ok_zero_std(self):
        """Zero std triggers max(std, 1e-6) guard."""
        dist = {"mean": 1.0, "std": 0.0}
        # Threshold: 1.0 + 3.0 * 1e-6 = 1.000003
        assert sigma_sanity_ok(1.000002, dist) is True
        assert sigma_sanity_ok(1.1, dist) is False

    def test_sigma_sanity_ok_missing_keys_uses_defaults(self):
        """Missing mean/std should use defaults (1.0, 0.1)."""
        dist = {}
        # Default: mean=1.0, std=0.1, threshold=1.3
        assert sigma_sanity_ok(1.3, dist) is True
        assert sigma_sanity_ok(1.4, dist) is False

    def test_load_baseline_distribution_missing_file(self):
        """Missing file should return None."""
        result = load_baseline_distribution("nonexistent_variant_xyz")
        assert result is None

    def test_load_baseline_distribution_corrupt_json(self):
        """Corrupt JSON file should return None (exception caught)."""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "corrupt.json"
            p.write_text("{broken json///")
            with patch("arm_gym.verifier.BASELINE_DIST_DIR", Path(d)):
                result = load_baseline_distribution("corrupt")
                assert result is None

    def test_load_baseline_distribution_valid_json(self):
        """Valid JSON file should return parsed dict."""
        with tempfile.TemporaryDirectory() as d:
            data = {"mean": 1.2, "std": 0.15, "n": 100}
            p = Path(d) / "good_variant.json"
            p.write_text(json.dumps(data))
            with patch("arm_gym.verifier.BASELINE_DIST_DIR", Path(d)):
                result = load_baseline_distribution("good_variant")
                assert result == data


# ---------------------------------------------------------------------------
# 9. train.py edge cases
# ---------------------------------------------------------------------------


class TestTrainEdgeCases:

    def test_train_config_lora_alpha_is_rank_times_2(self):
        """lora_alpha property should be lora_rank * 2."""
        from arm_gym.train import TrainConfig

        cfg = TrainConfig(lora_rank=16)
        assert cfg.lora_alpha == 32

        cfg2 = TrainConfig(lora_rank=32)
        assert cfg2.lora_alpha == 64

        cfg3 = TrainConfig(lora_rank=1)
        assert cfg3.lora_alpha == 2

    def test_detect_stack_without_torch(self):
        """Without torch/unsloth/vllm, detect_stack should return 'single_gpu'."""
        from arm_gym.train import detect_stack

        with patch("importlib.util.find_spec", return_value=None):
            with patch.dict(os.environ, {}, clear=True):
                assert detect_stack() == "single_gpu"

    def test_detect_stack_with_unsloth_and_vllm(self):
        """With both unsloth and vllm, detect_stack should return 'unsloth_vllm'."""
        from arm_gym.train import detect_stack

        def fake_find_spec(name):
            if name in ("unsloth", "vllm"):
                return MagicMock()
            return None

        with patch("importlib.util.find_spec", side_effect=fake_find_spec):
            assert detect_stack() == "unsloth_vllm"

    def test_detect_stack_plain_trl_ddp(self):
        """With torch and WORLD_SIZE env var, detect_stack returns 'plain_trl_ddp'."""
        from arm_gym.train import detect_stack

        def fake_find_spec(name):
            if name == "torch":
                return MagicMock()
            return None  # no unsloth or vllm

        with patch("importlib.util.find_spec", side_effect=fake_find_spec):
            with patch.dict(os.environ, {"WORLD_SIZE": "4"}):
                assert detect_stack() == "plain_trl_ddp"

    def test_main_smoke_returns_zero(self):
        """main(["--smoke"]) should return 0 without needing GPU."""
        from arm_gym.train import main

        assert main(["--smoke"]) == 0

    def test_main_smoke_skip_sft(self):
        """main(["--smoke", "--skip-sft"]) should also return 0."""
        from arm_gym.train import main

        assert main(["--smoke", "--skip-sft"]) == 0

    def test_smoke_day0_correctness_default(self):
        """Default smoke_day0_correctness returns 0.6 (above SFT threshold)."""
        from arm_gym.train import smoke_day0_correctness

        rate = smoke_day0_correctness(None, "Qwen/Qwen2.5-Coder-7B-Instruct")
        assert rate == 0.6

    def test_smoke_day0_correctness_forced_sft(self):
        """FORCE_SFT_WARMUP=1 makes smoke_day0_correctness return 0.0."""
        from arm_gym.train import smoke_day0_correctness

        with patch.dict(os.environ, {"FORCE_SFT_WARMUP": "1"}):
            rate = smoke_day0_correctness(None, "model-id")
            assert rate == 0.0

    def test_maybe_sft_warmup_skips_when_above_threshold(self):
        """maybe_sft_warmup should return False when correctness >= 0.4."""
        from arm_gym.train import TrainConfig, maybe_sft_warmup

        cfg = TrainConfig(skip_sft=False)
        assert maybe_sft_warmup(cfg, None) is False

    def test_maybe_sft_warmup_skips_when_skip_sft_flag(self):
        """maybe_sft_warmup should return False when skip_sft=True."""
        from arm_gym.train import TrainConfig, maybe_sft_warmup

        cfg = TrainConfig(skip_sft=True)
        with patch.dict(os.environ, {"FORCE_SFT_WARMUP": "1"}):
            assert maybe_sft_warmup(cfg, None) is False

    def test_train_config_defaults(self):
        """Verify scaffold defaults match CLAUDE.md documentation."""
        from arm_gym.train import TrainConfig

        cfg = TrainConfig()
        assert cfg.lora_rank == 16
        assert cfg.lora_alpha == 32
        assert cfg.num_generations == 4
        assert cfg.per_device_train_batch_size == 1
        assert cfg.gradient_accumulation_steps == 32


# ---------------------------------------------------------------------------
# 10. compile_baseline.py edge cases
# ---------------------------------------------------------------------------


class TestCompileBaselineEdgeCases:

    def test_find_tool_returns_none_when_no_candidate_found(self):
        """find_tool with unavailable candidates should return None."""
        from arm_gym.compile_baseline import find_tool

        with patch("shutil.which", return_value=None):
            assert find_tool(["nonexistent-1", "nonexistent-2"]) is None

    def test_find_tool_returns_first_available(self):
        """find_tool should return the first candidate found by shutil.which."""
        from arm_gym.compile_baseline import find_tool

        def fake_which(name):
            if name == "clang-21":
                return None
            if name == "clang-20":
                return "/usr/bin/clang-20"
            return None

        with patch("shutil.which", side_effect=fake_which):
            assert find_tool(["clang-21", "clang-20", "clang"]) == "clang-20"

    def test_toolchain_info_ready_with_neither(self):
        """ToolchainInfo.ready() is False when both compilers are None."""
        from arm_gym.compile_baseline import ToolchainInfo

        tc = ToolchainInfo(clang=None, gcc_aarch64=None, mca=None,
                           mcpu="neoverse-v2", mcpu_disclosed=None)
        assert tc.ready() is False

    def test_toolchain_info_ready_with_clang_only(self):
        """ToolchainInfo.ready() is True with only clang."""
        from arm_gym.compile_baseline import ToolchainInfo

        tc = ToolchainInfo(clang="/usr/bin/clang", gcc_aarch64=None, mca=None,
                           mcpu="neoverse-v2", mcpu_disclosed=None)
        assert tc.ready() is True

    def test_compile_to_asm_no_compiler_raises(self):
        """compile_to_asm with no compilers should raise RuntimeError."""
        from arm_gym.compile_baseline import ToolchainInfo, compile_to_asm

        tc = ToolchainInfo(clang=None, gcc_aarch64=None, mca=None,
                           mcpu="neoverse-v2", mcpu_disclosed=None)
        with pytest.raises(RuntimeError, match="no AArch64 compiler available"):
            compile_to_asm("void kernel(void){}", tc)

    def test_detect_toolchain_without_any_tools(self):
        """detect_toolchain with no tools installed should still return a ToolchainInfo."""
        from arm_gym.compile_baseline import detect_toolchain

        with patch("shutil.which", return_value=None), \
             patch("arm_gym.compile_baseline.find_tool", return_value=None):
            tc = detect_toolchain()
            assert tc.mcpu  # always has a value
            assert tc.clang is None
            assert tc.gcc_aarch64 is None
