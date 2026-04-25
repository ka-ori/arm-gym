from arm_gym.errors import ErrorKind, StructuredError, VerifierResult
from arm_gym.reward import RewardConfig, batch_rewards, group_zscore_clip, raw_reward


def _good(speedup: float, stalls=0, p99=0.5) -> VerifierResult:
    return VerifierResult(ok=True, reward=0.0, speedup=speedup,
                          agent_cycles=100, baseline_cycles=100 * speedup,
                          mca_dispatch_stalls=stalls, mca_resource_pressure_p99=p99)


def _bad() -> VerifierResult:
    return VerifierResult(ok=False, reward=0.0,
                          error=StructuredError(ErrorKind.ASSEMBLE_FAIL, "syntax"))


def test_failed_result_gets_zero_reward():
    r = raw_reward(_bad(), RewardConfig())
    assert r == 0.0


def test_binary_plus_speedup_default_mode():
    cfg = RewardConfig()  # binary_plus_speedup default
    assert raw_reward(_good(1.5), cfg) == 0.5
    assert raw_reward(_good(1.0), cfg) == 0.0  # no bonus at parity


def test_speedup_clipped_to_max():
    cfg = RewardConfig()
    # lucky 5x rollout cannot dominate — Cut 4 clip
    assert raw_reward(_good(5.0), cfg) == 2.0
    assert raw_reward(_good(10.0), cfg) == 2.0


def test_speedup_clipped_to_min():
    cfg = RewardConfig()
    # slowdowns floor at 1.0 → reward floor 0
    assert raw_reward(_good(0.5), cfg) == 0.0


def test_shaped_mode_requires_precise_hazard_def():
    cfg = RewardConfig(mode="shaped")
    # precise def: stalls==0 AND p99<1.0
    r_ok = raw_reward(_good(1.2, stalls=0, p99=0.5), cfg)
    r_stall = raw_reward(_good(1.2, stalls=5, p99=0.5), cfg)
    r_pressure = raw_reward(_good(1.2, stalls=0, p99=2.0), cfg)
    assert r_ok > r_stall
    assert r_ok > r_pressure


def test_shaped_mode_neon_requires_liveness_aware_check():
    """Dead NEON stores should NOT get the +0.05. Cut 3."""
    cfg = RewardConfig(mode="shaped")
    asm_dead = """
    .text
    fmov v0.4s, #1.0
    ret
    """
    asm_live = """
    .text
    fmov v0.4s, #1.0
    st1 {v0.4s}, [x0]
    ret
    """
    r_dead = raw_reward(_good(1.2), cfg, asm=asm_dead)
    r_live = raw_reward(_good(1.2), cfg, asm=asm_live)
    assert r_live > r_dead


def test_group_zscore_clip_bounds():
    out = group_zscore_clip([0.0, 0.1, 5.0, 0.05], clip=1.5)
    assert all(-1.5 <= r <= 1.5 for r in out)


def test_batch_rewards_z_score_tames_outlier():
    results = [_good(1.1), _good(1.2), _good(5.0), _good(1.0)]
    rs = batch_rewards(results, RewardConfig())
    assert max(rs) <= 1.5
    assert min(rs) >= -1.5
