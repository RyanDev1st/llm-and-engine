from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InvariantFailure:
    invariant_id: str
    turn_index: int
    reason: str


def validate_mode_invariants(messages: list[dict[str, Any]]) -> list[InvariantFailure]:
    failures: list[InvariantFailure] = []
    for idx, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role == "tool" and (idx == 0 or messages[idx - 1].get("role") != "assistant"):
            failures.append(InvariantFailure("INV_TOOL_ROLE_PRECEDENCE", idx, "tool must follow assistant"))
        if role != "assistant" or not isinstance(content, dict):
            continue
        msg_type = content.get("type")
        if msg_type == "tool_call" and "text" in content:
            failures.append(InvariantFailure("INV_TOOL_CALL_TURN_PURITY", idx, "tool call contains narration"))
        if idx > 0 and messages[idx - 1].get("role") == "tool" and msg_type != "narration_reply":
            failures.append(InvariantFailure("INV_POST_TOOL_NARRATION_PURITY", idx, "post-tool assistant must narrate"))
        if msg_type == "direct_reply" and idx + 1 < len(messages) and messages[idx + 1].get("role") == "tool":
            failures.append(InvariantFailure("INV_ROUTER_DIRECT_TERMINATES", idx, "direct reply followed by tool"))
    return failures
