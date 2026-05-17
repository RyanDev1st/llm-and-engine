from __future__ import annotations

from dataclasses import dataclass

from ..contracts.tool_grammar import parse_tool_name

EXPECTED_NONTOOL_SLICES = {"J", "K"}
EXPECTED_TOOL_SLICES = {"A", "B", "C", "D", "E", "F", "G", "H", "I"}


@dataclass(frozen=True)
class RoutingSanityFailure:
    rule_id: str
    reason: str


def check_routing_sanity(slice_name: str, messages: list[dict[str, str]]) -> list[RoutingSanityFailure]:
    failures: list[RoutingSanityFailure] = []
    tool_calls = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        if parse_tool_name(message.get("content", "")):
            tool_calls += 1

    if slice_name in EXPECTED_NONTOOL_SLICES and tool_calls != 0:
        failures.append(RoutingSanityFailure("V4_NONTOOL_SLICE", "J/K must contain zero tool calls"))
    if slice_name in EXPECTED_TOOL_SLICES and tool_calls == 0:
        failures.append(RoutingSanityFailure("V4_TOOL_SLICE", "A-I must contain at least one tool call"))
    return failures
