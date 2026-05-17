from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .contracts import validate_mode_invariants
from .engine_protocol import ToolBackend, ToolContext
from .grounding import validate_narration
from .json_outputs import parse_narrator_output, parse_router_output
from .phase_payloads import Message, NarratorPayload, RouterPayload, validate_narrator_payload, validate_router_payload


class ModelBackend(Protocol):
    def generate(self, phase: str, payload: object) -> str:
        ...


@dataclass(frozen=True)
class TurnResult:
    messages: list[dict]
    failures: list[str]


def run_turn(history: list[dict], summary: str, user_message: str, model: ModelBackend, tools: ToolBackend, conversation_id: str = "default") -> TurnResult:
    router_payload = RouterPayload([Message(**m) for m in history], summary, user_message)
    failures = [item.error_id for item in validate_router_payload(router_payload)]
    if failures:
        return TurnResult(history, failures)
    router = parse_router_output(model.generate("router", router_payload))
    if not router.ok:
        return TurnResult(history, [item.error_id for item in router.violations])
    assert router.payload is not None
    base = [*history, {"role": "user", "content": user_message}, {"role": "assistant", "content": router.payload}]
    if router.payload["type"] == "direct_reply":
        return TurnResult(base, [item.invariant_id for item in validate_mode_invariants(base)])
    tool_result = tools.execute(router.payload["tool"], router.payload["args"], ToolContext(conversation_id))
    with_tool = [*base, {"role": "tool", "content": tool_result}]
    narrator_payload = NarratorPayload([Message(**m) for m in with_tool], summary, tool_result)
    payload_failures = [item.error_id for item in validate_narrator_payload(narrator_payload)]
    if payload_failures:
        return TurnResult(with_tool, payload_failures)
    narrator = parse_narrator_output(model.generate("narrator", narrator_payload))
    if not narrator.ok:
        return TurnResult(with_tool, [item.error_id for item in narrator.violations])
    assert narrator.payload is not None
    grounding = [item.error_id for item in validate_narration(narrator.payload["text"], tool_result)]
    final = [*with_tool, {"role": "assistant", "content": narrator.payload}]
    invariants = [item.invariant_id for item in validate_mode_invariants(final)]
    return TurnResult(final, [*grounding, *invariants])
