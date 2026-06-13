from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import chess

from .contracts import MAX_TOOL_CALLS, REQUIRED_FIELDS, RULES, SLICES, VALID_ROLES

# A tool call may be preceded by a short lead-in sentence in the same assistant
# turn (conversational shape), so we SEARCH rather than anchor.
_CALL = re.compile(r"<tool>\s*([a-z_][a-z0-9_]*)(.*?)</tool>", re.DOTALL)
_ARG = re.compile(r"([a-z_][a-z0-9_]*)=([^\s<>]+)")
_MOVE_SAN = re.compile(r"<tool>\s*move\s+san=([^\s<]+)")
# "Facts" the narration must copy from the tool result: eval/delta numbers and
# SAN moves. Used by the narration-grounding check (and mirrors the loss-weight
# target on the training side).
_FACT = re.compile(r"[+-]?\d+\.\d{2}|O-O(?:-O)?|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?")


_SKILL = re.compile(r"<skill>\s*([A-Za-z0-9_][A-Za-z0-9_-]*)\s*</skill>")


def _tool_matches(content: str) -> list[re.Match]:
    return list(_CALL.finditer(content))


def _skill_loads(messages: list[dict[str, str]]) -> list[str]:
    """Skill names loaded via the <skill>NAME</skill> verb, in order."""
    return [m for msg in messages if msg.get("role") == "assistant"
            for m in _SKILL.findall(msg.get("content", ""))]


def _actions(messages: list[dict[str, str]]) -> list[tuple[str, str]]:
    """Ordered (kind, name) stream of harness actions: ('skill'|'tool', name).
    One per assistant message by contract; preserves cross-message order."""
    out: list[tuple[str, str]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        for s in _SKILL.findall(content):
            out.append(("skill", s))
        for match in _tool_matches(content):
            out.append(("tool", match.group(1)))
    return out


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
    violations.extend(_skill_index_order(row, calls))
    violations.extend(_skill_body(row, calls, tools))
    violations.extend(_applies_when(row, calls))
    violations.extend(_plugin_only(row, calls))
    violations.extend(_grounding(row, calls))
    violations.extend(_engine_grounded(row))
    violations.extend(_eval_language(row))
    violations.extend(_narration_grounded(row))
    violations.extend(_injection(row))
    violations.extend(_move_legality(row))
    violations.extend(_board_state_turn(row))
    violations.extend(_one_tool_per_message(row["messages"]))
    violations.extend(_reasoning_mode(row))
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
        for match in _tool_matches(message.get("content", "")):
            calls.append((match.group(1), dict(_ARG.findall(match.group(2))), match.group(0)))
    return calls


def _skills(row: dict[str, Any]) -> list[Violation]:
    skills = {skill.get("name"): skill for skill in row["skills_index"]}
    enabled = set(row.get("plugin_context", {}).get("enabled", []))
    out = []
    for selected in row["selected_skills"]:
        skill = skills.get(selected)
        if not skill:
            out.append(Violation("selected_skill_exists", selected))
        elif skill.get("plugin") and (not skill.get("enabled", True) or skill.get("plugin") not in enabled):
            out.append(Violation("selected_skill_exists", selected))
    loaded = _skill_loads(row["messages"])
    for name in loaded:                       # every <skill> must be a listed skill
        if name not in skills:
            out.append(Violation("selected_skill_exists", f"unknown skill <skill>{name}</skill>"))
    for selected in row["selected_skills"]:
        if selected not in loaded:
            out.append(Violation("skill_loaded_after_selection", selected))
    return out


def _is_action(content: str) -> bool:
    return bool(_tool_matches(content)) or bool(_SKILL.search(content))


def _final(messages: list[dict[str, str]]) -> list[Violation]:
    finals = [m["content"] for m in messages if m.get("role") == "assistant" and not _is_action(m.get("content", ""))]
    if not finals:
        return [Violation("final_no_xml", "missing final assistant answer")]
    final = finals[-1]
    leaks = any(t in final for t in ("<tool>", "</tool>", "<skill>", "</skill>"))
    return [Violation("final_no_xml", "final contains raw action XML")] if leaks else []


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


def _skill_index_order(row: dict[str, Any], calls: list[tuple[str, dict[str, str], str]]) -> list[Violation]:
    if "skill_index_only_before_load" not in row.get("acceptance_rules", []):
        return []
    indexed = {skill.get("name") for skill in row.get("skills_index", [])}
    loaded: set[str] = set()
    out: list[Violation] = []
    for kind, name in _actions(row["messages"]):
        if kind == "skill":
            if name in indexed:
                loaded.add(name)
            continue
        if name == "normalize_human_chat" and "hood-human-chat" in indexed and "hood-human-chat" not in loaded:
            out.append(Violation("skill_index_only_before_load", "helper tool before helper skill load"))
    return out


def _skill_body(
    row: dict[str, Any], calls: list[tuple[str, dict[str, str], str]], tools: dict[str, Any]
) -> list[Violation]:
    if "skill_body_strict" not in row.get("acceptance_rules", []):
        return []
    out: list[Violation] = []
    selected = set(row.get("selected_skills", []))
    loaded: set[str] = set()
    for kind, name in _actions(row["messages"]):
        if kind == "skill":
            if name not in selected:
                out.append(Violation("skill_body_strict", f"irrelevant skill loaded: {name}"))
            loaded.add(name)
            continue
        tool = tools.get(name, {})
        if name == "normalize_human_chat" and "hood-human-chat" in selected and "hood-human-chat" not in loaded:
            out.append(Violation("skill_body_strict", "helper tool before helper skill load"))
        if tool.get("plugin") == "user-skills" and not (loaded & selected):
            out.append(Violation("skill_body_strict", f"tool before selected skill load: {name}"))
    return out


def _engine_grounded(row: dict[str, Any]) -> list[Violation]:
    if "engine_grounded" not in row.get("acceptance_rules", []):
        return []
    text = "\n".join(m.get("content", "") for m in row.get("messages", [])).lower()
    if "stockfish" not in text and "eval:" not in text and "score:" not in text:
        return [Violation("engine_grounded", "missing engine evidence")]
    return []


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


def _narration_grounded(row: dict[str, Any]) -> list[Violation]:
    """Anti-fabrication: any concrete value the final reply STATES — a pawn
    number or a SAN move — must appear in a tool result, so the model copies
    rather than invents. A purely qualitative final (no such value) is allowed:
    the default coaching reply describes the standing in words, by design. Sign
    prefixes are normalized so '+4.47' (tool) grounds '4.47' (reply)."""
    if "narration_grounded" not in row.get("acceptance_rules", []):
        return []
    messages = row["messages"]

    def facts(text: str) -> set[str]:
        # <think> is intent/plan reasoning, never facts — strip it so the grounding
        # check ignores it (the trained thinking trace must not be fact-checked).
        text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
        return {f.lstrip("+-") for f in _FACT.findall(text)}

    tool_facts: set[str] = set()
    for m in messages:
        if m.get("role") == "tool":
            tool_facts |= facts(m.get("content", ""))
    finals = [m["content"] for m in messages
              if m.get("role") == "assistant" and not _is_action(m.get("content", ""))]
    final_facts = facts(finals[-1]) if finals else set()
    if final_facts <= tool_facts:
        return []
    return [Violation("narration_grounded", "final cites a value absent from the tool results")]


def _applies_when(row: dict[str, Any], calls: list[tuple[str, dict[str, str], str]]) -> list[Violation]:
    tools = {tool["name"]: tool for tool in row["tool_manifest"]}
    out: list[Violation] = []
    history = _has_move_history(row["messages"])
    for name, _, raw in calls:
        applies = tools.get(name, {}).get("applies_when", "always")
        if applies == "has_history" and not history:
            out.append(Violation("applies_when_respected", f"{name} needs prior move"))
    return out


def _has_move_history(messages: list[dict[str, str]]) -> bool:
    for message in messages:
        if message.get("role") != "tool":
            continue
        text = message.get("content", "").lower()
        if ("move:" in text and "success" in text) or text.startswith("success:"):
            return True
    return False


def _plugin_only(row: dict[str, Any], calls: list[tuple[str, dict[str, str], str]]) -> list[Violation]:
    tools = {tool["name"]: tool for tool in row["tool_manifest"]}
    enabled = set(row.get("plugin_context", {}).get("enabled", []))
    out = []
    for name, _, _ in calls:
        tool = tools.get(name)
        if not tool:
            out.append(Violation("plugin_only_tools", name))
        elif tool.get("plugin") and (not tool.get("enabled", True) or tool.get("plugin") not in enabled):
            out.append(Violation("plugin_only_tools", name))
    return out


def _move_legality(row: dict[str, Any]) -> list[Violation]:
    """Every `move san=X` must be legal in position_fen, replayed in order."""
    fen = row.get("position_fen")
    if not fen:
        return []
    try:
        board = chess.Board(fen)
    except ValueError:
        return [Violation("illegal_move", "unparseable position_fen")]
    out: list[Violation] = []
    for message in row.get("messages", []):
        if message.get("role") != "assistant":
            continue
        for san in _MOVE_SAN.findall(message.get("content", "")):
            try:
                board.push(board.parse_san(san))
            except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError, ValueError):
                out.append(Violation("illegal_move", f"{san} illegal in {board.fen()}"))
                return out
    return out


def _board_state_turn(row: dict[str, Any]) -> list[Violation]:
    fen = row.get("position_fen")
    if not fen:
        return []
    side = "white" if fen.split()[1] == "w" else "black"
    out: list[Violation] = []
    for message in row.get("messages", []):
        if message.get("role") != "tool":
            continue
        content = message.get("content", "")
        if content.startswith("board_state:") and "turn=" in content and f"turn={side}" not in content:
            out.append(Violation("board_state_grounded", "board_state turn != FEN side"))
    return out


def _one_tool_per_message(messages: list[dict[str, str]]) -> list[Violation]:
    """One tool call per inference step: each assistant message holds at most one
    `<tool>` call (a lead-in sentence may precede it). Many calls across the loop
    are fine — they live in separate assistant messages."""
    out: list[Violation] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content", "")
        if len(_tool_matches(content)) + len(_SKILL.findall(content)) > 1:
            out.append(Violation("one_tool_per_message", "multiple actions in one inference step"))
    return out


def _reasoning_mode(row: dict[str, Any]) -> list[Violation]:
    """Dual-mode integrity: a `fast` row must carry NO <think> (else the model
    can never run snappy); `think`/`auto`/unset are unconstrained here. This is
    what makes fast-vs-think a real toggle rather than an always-on reflex."""
    mode = (row.get("reasoning_mode") or "").strip().lower()
    if mode != "fast":
        return []
    for m in row.get("messages", []):
        if m.get("role") == "assistant" and "<think>" in m.get("content", ""):
            return [Violation("reasoning_mode_fast_no_think", "fast row contains <think>")]
    return []


def _injection(row: dict[str, Any]) -> list[Violation]:
    if "tool_text_is_data" not in row["acceptance_rules"]:
        return []
    finals = [m.get("content", "").lower() for m in row["messages"] if m.get("role") == "assistant"]
    final = finals[-1] if finals else ""
    bad = ("ignore previous" in final) or ("system overridden" in final)
    return [Violation("tool_text_is_data", "followed injected text")] if bad else []
