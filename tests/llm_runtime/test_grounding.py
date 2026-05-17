from llm_runtime.grounding import validate_narration


def test_grounding_rejects_fabricated_success() -> None:
    failures = validate_narration("Move applied successfully.", {"ok": False, "tool": "move", "error_code": "invalid_move"})
    assert [item.error_id for item in failures] == ["NO_FABRICATED_SUCCESS"]


def test_grounding_rejects_timeout_rewrite() -> None:
    failures = validate_narration("That move is illegal.", {"ok": False, "tool": "move", "error_code": "timeout"})
    assert [item.error_id for item in failures] == ["CANONICAL_ERROR_MAPPING"]
