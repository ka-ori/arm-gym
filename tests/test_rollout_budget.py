from arm_gym.rollout_budget import (
    DEFAULT_N_TESTS,
    TestCase,
    run_parallel,
    select_adversarial,
    update_adversarial_ranks,
)


def test_default_n_is_20_per_weaker_flag_3():
    assert DEFAULT_N_TESTS == 20


def test_select_adversarial_ranks_higher_first():
    tests = [TestCase((), None, adversarial_rank=r) for r in [0.1, 5.0, 1.0, 3.0]]
    picked = select_adversarial(tests, n=2)
    assert [t.adversarial_rank for t in picked] == [5.0, 3.0]


def test_make_edge_case_tests_high_rank():
    from arm_gym.rollout_budget import EDGE_CASES, make_edge_case_tests
    ec = make_edge_case_tests()
    assert len(ec) == len(EDGE_CASES)
    assert all(t.adversarial_rank == 10.0 for t in ec)
    assert {t.inputs[0] for t in ec if t.inputs[0] == t.inputs[0]} >= {0.0, 1.0, -1.0}


def test_run_parallel_short_circuits_on_failure():
    calls = {"n": 0}

    def runner(t):
        calls["n"] += 1
        return t.expected is True

    tests = [TestCase((), False)] + [TestCase((), True)] * 10
    ok, idx = run_parallel(runner, tests)
    assert ok is False
    assert 0 <= idx < len(tests)


def test_run_parallel_all_pass():
    tests = [TestCase((), True) for _ in range(5)]
    ok, idx = run_parallel(lambda t: True, tests)
    assert ok is True
    assert idx == -1


def test_adversarial_rank_ema_decays_and_boosts():
    tests = [TestCase((), None, adversarial_rank=r) for r in [1.0, 1.0, 1.0]]
    update_adversarial_ranks(tests, failures_by_index=[1])
    assert tests[1].adversarial_rank > tests[0].adversarial_rank
    # non-failing ranks decayed
    assert tests[0].adversarial_rank < 1.0
