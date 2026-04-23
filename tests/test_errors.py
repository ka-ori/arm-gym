import json

from arm_gym.errors import ErrorKind, StructuredError, VerifierResult


def test_structured_error_round_trips_to_json_for_prompt():
    e = StructuredError(kind=ErrorKind.ASSEMBLE_FAIL, message="bad token",
                        line=3, column=5, extra={"token": "xzrq"})
    payload = json.loads(e.to_prompt())
    assert payload["kind"] == "assemble_fail"
    assert payload["line"] == 3
    assert payload["column"] == 5
    assert payload["extra"]["token"] == "xzrq"


def test_empty_fields_omitted():
    e = StructuredError(kind=ErrorKind.SEGFAULT, message="boom")
    payload = json.loads(e.to_prompt())
    assert "line" not in payload
    assert "extra" not in payload


def test_verifier_result_defaults_fail_closed():
    v = VerifierResult(ok=False, reward=0.0)
    assert v.ok is False
    assert v.reward == 0.0
    assert v.speedup is None
