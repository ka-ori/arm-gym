from arm_gym.kernels import TEMPLATES, generate_all, generate_variants, split_train_eval, summary


def test_template_count_matches_audit_target():
    assert 15 <= len(TEMPLATES) <= 20, "Cut 2 target: 15-20 templates"


def test_variants_reach_500_plus():
    all_v = generate_all()
    assert len(all_v) >= 500, f"Cut 2 target: 500-2000 variants, got {len(all_v)}"


def test_variant_ids_are_unique():
    ids = [v.variant_id for v in generate_all()]
    assert len(ids) == len(set(ids))


def test_vec_add_renders_valid_c():
    v = next(generate_variants("vec_add"))
    assert "#include <stddef.h>" in v.c_source
    assert "kernel" in v.c_source
    assert "for" in v.c_source


def test_split_is_deterministic_and_disjoint():
    vs = generate_all()
    t1, e1 = split_train_eval(vs, eval_frac=0.1, seed=42)
    t2, e2 = split_train_eval(vs, eval_frac=0.1, seed=42)
    assert {v.variant_id for v in t1} == {v.variant_id for v in t2}
    assert {v.variant_id for v in e1} == {v.variant_id for v in e2}
    overlap = {v.variant_id for v in t1} & {v.variant_id for v in e1}
    assert overlap == set()


def test_summary_shape():
    s = summary()
    assert s["templates"] == len(TEMPLATES)
    assert s["variants"] >= 500
