from arm_gym.mca import parse_mca, uses_neon_with_liveness


def test_parse_mca_basic_shape():
    out = """Iterations:        100
Instructions:      12
Total Cycles:      7
Total uOps:        12

Dispatch Width Stalls: 0
IPC:               1.71

Resource pressure per iteration:
[0]   [1]   [2]
0.50  0.20  0.10
"""
    rep = parse_mca(out)
    assert rep.total_cycles == 7
    assert rep.instructions == 12
    assert abs(rep.ipc - 1.71) < 0.01
    assert rep.dispatch_stalls == 0
    # p99 among {0.5, 0.2, 0.1}
    assert rep.resource_pressure_p99 <= 0.5
    assert rep.no_pipeline_hazard is True


def test_hazard_blocked_by_stalls():
    out = """Instructions:      10
Total Cycles:      10
IPC:               1.0
Dispatch Width Stalls: 4
Resource pressure per iteration:
[0]
0.1
"""
    rep = parse_mca(out)
    assert rep.no_pipeline_hazard is False


def test_hazard_blocked_by_pressure():
    out = """Instructions:      10
Total Cycles:      10
IPC:               1.0
Dispatch Width Stalls: 0
Resource pressure per iteration:
[0]
2.5
"""
    rep = parse_mca(out)
    assert rep.no_pipeline_hazard is False


def test_neon_liveness_dead_store():
    dead = """
    fmov v0.4s, #1.0
    ret
    """
    assert uses_neon_with_liveness(dead) is False


def test_neon_liveness_live_store():
    live = """
    fmov v0.4s, #1.0
    str q0, [x0]
    ret
    """
    assert uses_neon_with_liveness(live) is True


def test_neon_liveness_chain_to_non_neon():
    """NEON register consumed before ret counts as live."""
    asm = """
    ld1 {v0.4s}, [x1]
    fadd v0.4s, v0.4s, v0.4s
    st1 {v0.4s}, [x0]
    ret
    """
    assert uses_neon_with_liveness(asm) is True
