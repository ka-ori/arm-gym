"""Kaggle reward sanity: non-zero signal without HuggingFace dataset deps."""


def test_verify_rewards_main_succeeds():
    from kaggle import verify_rewards

    verify_rewards.main()
