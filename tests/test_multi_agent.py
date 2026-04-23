from arm_gym.multi_agent import (analyzer_credit, build_optimizer_prompt,
                                 parse_analyzer_output, per_role_reward, OpportunityToken)


def test_parses_strict_json_array():
    out = '[{"kind":"vectorize_neon","target_region":"loop","rationale":"wide"}]'
    toks = parse_analyzer_output(out)
    assert len(toks) == 1
    assert toks[0].kind == "vectorize_neon"


def test_parses_json_embedded_in_prose():
    out = "thinking... [{\"kind\":\"use_ldp_stp\"}]   done"
    toks = parse_analyzer_output(out)
    assert toks and toks[0].kind == "use_ldp_stp"


def test_ignores_garbage():
    assert parse_analyzer_output("no json here") == []
    assert parse_analyzer_output("[not, valid, json]") == []


def test_optimizer_prompt_contains_all_sections():
    prompt = build_optimizer_prompt(
        "int k(){return 0;}",
        ".text\nret",
        [OpportunityToken("vectorize_neon")],
    )
    assert "C Code:" in prompt
    assert "Baseline Assembly" in prompt
    assert "Opportunity Tokens:" in prompt
    assert "vectorize_neon" in prompt
    assert "<assembly>" in prompt


def test_analyzer_credit_when_suggestion_applied():
    toks = [OpportunityToken("use_ldp_stp")]
    asm_with = "ldp x0, x1, [sp]\nret"
    asm_without = "ldr x0, [sp]\nret"
    assert analyzer_credit(toks, asm_with) == 1.0
    assert analyzer_credit(toks, asm_without) == 0.0


def test_per_role_reward_splits():
    toks = [OpportunityToken("fuse_multiply_add")]
    asm = "fmadd s0, s1, s2, s3\nret"
    split = per_role_reward(1.0, toks, asm)
    assert split["optimizer"] == 1.0
    assert split["analyzer"] > 0.0


def test_per_role_reward_zero_when_suggestions_unused():
    toks = [OpportunityToken("fuse_multiply_add")]
    asm = "ldr x0, [sp]\nret"
    split = per_role_reward(1.0, toks, asm)
    assert split["analyzer"] == 0.0
