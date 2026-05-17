from llm_runtime.json_outputs import parse_narrator_output, parse_router_output


def test_router_accepts_tool_call() -> None:
    parsed = parse_router_output('{"type":"tool_call","tool":"best_move","args":{}}')
    assert parsed.ok is True


def test_router_rejects_extra_text() -> None:
    parsed = parse_router_output('hi {"type":"tool_call","tool":"best_move","args":{}}')
    assert [item.error_id for item in parsed.violations] == ["JSON_PARSE_FAILED"]


def test_narrator_rejects_tool_leakage() -> None:
    parsed = parse_narrator_output('{"type":"narration_reply","text":"ok","tool":"eval"}')
    assert any(item.error_id == "NARRATOR_SCHEMA_INVALID" for item in parsed.violations)
