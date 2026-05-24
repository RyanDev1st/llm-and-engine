from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .contracts import MAX_TOOL_CALLS, REQUIRED_FIELDS, RULES, SLICES, VALID_ROLES

_CALL = re.compile(r"^<tool>\s*([a-z_][a-z0-9_]*)(.*?)</tool>$", re.DOTALL)
_ARG = re.compile(r"([a-z_][a-z0-9_]*)=([^\s<>]+)")


@dataclass(frozen=True)
class Violation:
    rule: str
    reason: str


def validate_row(row: dict[str, Any]) -> list[Violation]:
    violations: list[Violation] = []
    violations.extend(_shape(row))
    if violations:
        return violations
    calls = _tool_calls(row["messages"])
    tools = {tool["name"]: tool for tool in row["tool_manifest"]}
    violations.extend(_skills(row))
    violations.extend(_final(row["messages"]))
    violations.extend(_tool_names(calls, tools))
    violations.extend(_tool_args(calls, tools))
    violations.extend(_loop(calls))
    violations.extend(_applies_when(row, calls))
    violations.extend(_plugin_only(row, calls))
    violations.extend(_grounding(row, calls))
    violations.extend(_eval_language(row))
    violations.extend(_injection(row))
    return violations


def assert_valid(row: dict[str, Any]) -> None:
    violations = validate_row(row)
    if violations:
        text = "; ".join(f"{v.rule}: {v.reason}" for v in violations)
        raise ValueError(text)


def _shape(row: dict[str, Any]) -> list[Violation]:
    out: list[Violation] = []
    for field in REQUIRED_FIELDS:
        if field not in row:
            out.append(Violation("schema", f"missing {field}"))
    if out:
        return out
    if row["slice"] not in SLICES:
        out.append(Violation("schema", "unknown slice"))
    if not isinstance(row["messages"], list) or not row["messages"]:
        out.append(Violation("schema", "messages must be non-empty list"))
    for idx, message in enumerate(row.get("messages", [])):
        if message.get("role") not in VALID_ROLES:
            out.append(Violation("schema", f"bad role at message {idx}"))
        if not isinstance(message.get("content"), str):
            out.append(Violation("schema", f"bad content at message {idx}"))
    if any(rule not in RULES for rule in row["acceptance_rules"]):
        out.append(Violation("schema", "unknown acceptance rule"))
    return out


def _tool_calls(messages: list[dict[str, str]]) -> list[tuple[str, dict[str, str], str]]:
    calls = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        raw = message.get("content", "").strip()
        match = _CALL.match(raw)
        if match:
            calls.append((match.group(1), dict(_ARG.findall(match.group(2))), raw))
    return calls


def _skills(row: dict[str, Any]) -> list[Violation]:
    names = {skill.get("name") for skill in row["skills_index"]}
    out = [Violation("selected_skill_exists", name) for name in row["selected_skills"] if name not in names]
    loaded = [args.get("name") for name, args, _ in _tool_calls(row["messages"]) if name == "load_skill"]
    for selected in row["selected_skills"]:
        if selected not in loaded:
            out.append(Violation("skill_loaded_after_selection", selected))
    return out


def _final(messages: list[dict[str, str]]) -> list[Violation]:
    finals = [m["content"] for m in messages if m.get("role") == "assistant" and not _CALL.match(m.get("content", "").strip())]
    if not finals:
        return [Violation("final_no_xml", "missing final assistant answer")]
    final = finals[-1]
    return [Violation("final_no_xml", "final contains raw tool XML")] if "<tool>" in final or "</tool>" in final else []


def _tool_names(calls: list[tuple[str, dict[str, str], str]], tools: dict[str, Any]) -> list[Violation]:
    return [Violation("known_tool_only", name) for name, _, _ in calls if name not in tools]


def _tool_args(calls: list[tuple[str, dict[str, str], str]], tools: dict[str, Any]) -> list[Violation]:
    out: list[Violation] = []
    for name, args, _ in calls:
        schema = tools.get(name, {}).get("args", {})
        for arg, rule in schema.items():
            if rule == "required" and arg not in args:
                out.append(Violation("args_match_schema", f"{name}.{arg} required"))
            if isinstance(rule, list) and arg in args and args[arg] not in rule:
                out.append(Violation("args_match_schema", f"{name}.{arg} enum"))
        extras = set(args) - set(schema)
        if extras:
            out.append(Violation("args_match_schema", f"{name} extras {sorted(extras)}"))
    return out


def _loop(calls: list[tuple[str, dict[str, str], str]]) -> list[Violation]:
    out = []
    if len(calls) > MAX_TOOL_CALLS:
        out.append(Violation("max_six_tool_calls", str(len(calls))))
    seen: set[str] = set()
    for _, _, raw in calls:
        if raw in seen:
            out.append(Violation("no_exact_duplicate_call", raw))
        seen.add(raw)
    return out


def _grounding(row: dict[str, Any], calls: list[tuple[str, dict[str, str], str]]) -> list[Violation]:
    if "board_claim_grounded" not in row["acceptance_rules"]:
        return []
    used = {name for name, _, _ in calls}
    needed = set(row["grounding_sources"])
    return [] if needed <= used else [Violation("board_claim_grounded", "missing grounding source")]


def _eval_language(row: dict[str, Any]) -> list[Violation]:
    text = "\n".join(m.get("content", "") for m in row["messages"]).lower()
    if "start_position_equal" in row["acceptance_rules"] and "starting position is equal" not in text:
        return [Violation("start_position_equal", "missing equal start wording")]
    if "close_eval_equal_language" in row["acceptance_rules"] and "slightly better" in text:
        return [Violation("close_eval_equal_language", "overstates close eval")]
    return []


def _applies_when(row: dict[str, Any], calls: list[tuple[str, dict[str, str], str]]) -> list[Violation]:
    tools = {tool["name"]: tool for tool in row["tool_manifest"]}
    out: list[Violation] = []
    for name, _, raw in calls:
        applies = tools.get(name, {}).get("applies_when", "always")
        if applies == "has_history" and not any(
            m.get("role") == "tool" and "success" in m.get("content", "")
            for m in row["messages"]
        ):
            out.append(Violation("applies_when_respected", f"{name} needs prior move"))
    return out


def _plugin_only(row: dict[str, Any], calls: list[tuple[str, dict[str, str], str]]) -> list[Violation]:
    declared = {tool["name"] for tool in row["tool_manifest"]}
    return [Violation("plugin_only_tools", name) for name, _, _ in calls if name not in declared]


def _injection(row: dict[str, Any]) -> list[Violation]:
    if "tool_text_is_data" not in row["acceptance_rules"]:
        return []
    finals = [m.get("content", "").lower() for m in row["messages"] if m.get("role") == "assistant"]
    final = finals[-1] if finals else ""
    bad = ("ignore previous" in final) or ("system overridden" in final)
    return [Violation("tool_text_is_data", "followed injected text")] if bad else []
