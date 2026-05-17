from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

ALLOWED_TOOLS = {"eval", "best_move", "review_move", "move", "legal_moves", "list_pieces", "undo", "threats", "ask_chessbot"}


@dataclass(frozen=True)
class OutputViolation:
    error_id: str
    reason: str


@dataclass(frozen=True)
class ParsedOutput:
    payload: dict[str, Any] | None
    violations: list[OutputViolation]

    @property
    def ok(self) -> bool:
        return not self.violations and self.payload is not None


def parse_router_output(text: str) -> ParsedOutput:
    parsed = _parse_exact_json_object(text)
    if parsed.violations:
        return parsed
    assert parsed.payload is not None
    return ParsedOutput(parsed.payload, _router_violations(parsed.payload))


def parse_narrator_output(text: str) -> ParsedOutput:
    parsed = _parse_exact_json_object(text)
    if parsed.violations:
        return parsed
    assert parsed.payload is not None
    return ParsedOutput(parsed.payload, _narrator_violations(parsed.payload))


def _parse_exact_json_object(text: str) -> ParsedOutput:
    stripped = text.strip()
    if stripped.startswith("```") or stripped.endswith("```"):
        return ParsedOutput(None, [OutputViolation("JSON_EXTRA_TEXT", "markdown fence not allowed")])
    decoder = json.JSONDecoder()
    try:
        payload, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        return ParsedOutput(None, [OutputViolation("JSON_PARSE_FAILED", exc.msg)])
    if stripped[end:].strip():
        return ParsedOutput(None, [OutputViolation("JSON_EXTRA_TEXT", "extra text after JSON")])
    if not isinstance(payload, dict):
        return ParsedOutput(None, [OutputViolation("JSON_NON_OBJECT", "output must be object")])
    return ParsedOutput(payload, [])


def _router_violations(payload: dict[str, Any]) -> list[OutputViolation]:
    output_type = payload.get("type")
    if output_type == "tool_call":
        expected = {"type", "tool", "args"}
        violations = _exact_keys(payload, expected, "ROUTER_SCHEMA_INVALID")
        if not isinstance(payload.get("tool"), str) or not payload.get("tool"):
            violations.append(OutputViolation("ROUTER_SCHEMA_INVALID", "tool must be non-empty string"))
        elif payload["tool"] not in ALLOWED_TOOLS:
            violations.append(OutputViolation("ROUTER_UNKNOWN_TOOL", f"unknown tool: {payload['tool']}"))
        if not isinstance(payload.get("args"), dict):
            violations.append(OutputViolation("ROUTER_SCHEMA_INVALID", "args must be object"))
        return violations
    if output_type == "direct_reply":
        violations = _exact_keys(payload, {"type", "text"}, "ROUTER_SCHEMA_INVALID")
        if not isinstance(payload.get("text"), str) or not payload.get("text").strip():
            violations.append(OutputViolation("ROUTER_SCHEMA_INVALID", "text must be non-empty string"))
        return violations
    if {"tool", "args", "text"} & set(payload):
        return [OutputViolation("ROUTER_MIXED_OUTPUT", "payload mixes router/narrator fields")]
    return [OutputViolation("ROUTER_SCHEMA_INVALID", "unknown router type")]


def _narrator_violations(payload: dict[str, Any]) -> list[OutputViolation]:
    violations = _exact_keys(payload, {"type", "text"}, "NARRATOR_SCHEMA_INVALID")
    if payload.get("type") != "narration_reply":
        violations.append(OutputViolation("NARRATOR_SCHEMA_INVALID", "type must be narration_reply"))
    if {"tool", "args"} & set(payload):
        violations.append(OutputViolation("NARRATOR_TOOL_LEAKAGE", "narrator output contains tool fields"))
    if not isinstance(payload.get("text"), str) or not payload.get("text", "").strip():
        violations.append(OutputViolation("NARRATOR_SCHEMA_INVALID", "text must be non-empty string"))
    return violations


def _exact_keys(payload: dict[str, Any], expected: set[str], error_id: str) -> list[OutputViolation]:
    actual = set(payload)
    if actual == expected:
        return []
    return [OutputViolation(error_id, f"expected keys {sorted(expected)}, got {sorted(actual)}")]
