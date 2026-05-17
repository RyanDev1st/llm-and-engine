from llm_runtime.phase_payloads import Message, NarratorPayload, RouterPayload, validate_narrator_payload, validate_router_payload


def test_router_payload_requires_summary() -> None:
    payload = RouterPayload([Message("system", "x")], "", "hello")
    assert [item.error_id for item in validate_router_payload(payload)] == ["INV_HISTORY_SUMMARY_REQUIRED"]


def test_narrator_requires_tool_role() -> None:
    payload = NarratorPayload([Message("assistant", {})], "sum", {"ok": True})
    assert [item.error_id for item in validate_narrator_payload(payload)] == ["INV_NARRATOR_REQUIRES_TOOL"]
