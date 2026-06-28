from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import chess

from .contracts import MAX_TOOL_CALLS, REQUIRED_FIELDS, RULES, SLICES, VALID_ROLES

# v5-native: tool calls are STRUCTURED on the assistant message
# (message["tool_calls"]), not <tool>…</tool> text. Loading a skill is the native
# tool call load_skill{name:NAME}. The grounding/legality/plan checks below still
# read tool RESULTS and FINAL text (both plain content), so only the call/skill
# accessors change.
LOAD_SKILL = "load_skill"

# "Facts" the narration must copy from the tool result: eval/delta numbers and
# SAN moves. Used by the narration-grounding check (and mirrors the loss-weight
# target on the training side).
_FACT = re.compile(r"[+-]?\d+\.\d{2}|O-O(?:-O)?|[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?")

# Standing/eval vocabulary a final uses to ASSERT a position fact (the text.eval_magnitude
# / score_phrase registers). A follow-up turn that states one of these must re-ground it
# with a tool call — never answer "same read as before" from memory (the v1-v4 confab gap).
_STANDING = re.compile(
    r"\b(on top|winning|crushing|ahead by|up by|better by|clearly better|clear margin|"
    r"clear edge|small edge|slight edge|big advantage|commanding|near-decisive|"
    r"all but decided|dead level|roughly balanced|anyone's game|slightly ahead|"
    r"a touch better|forced mate|mate in)\b",
    re.I,
)

# PLAN-mode structure (still emitted as text in the plan deliverable): <goal>…</goal>,
# <plan>…</plan>, and checkbox bindings (the trailing "(name)" the serve gate maps to
# the executed skill/tool).
_GOAL_TAG = re.compile(r"<goal>(.*?)</goal>", re.DOTALL)
_PLAN_TAG = re.compile(r"<plan>(.*?)</plan>", re.DOTALL)
_BOX_BIND = re.compile(r"-\s*\[[ xX]\]\s*.+?\(([^)]+)\)")


def _msg_calls(msg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Structured (name, args) calls on one assistant message."""
    out: list[tuple[str, dict[str, Any]]] = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", tc)
        out.append((fn.get("name", ""), dict(fn.get("arguments", {}) or {})))
    return out


def _canonical(name: str, args: dict[str, Any]) -> str:
    """A stable string for a call (dedup + legality), order-independent over args."""
    body = ",".join(f"{k}={args[k]}" for k in sorted(args))
    return f"{name}({body})"


def _is_action(msg: dict[str, Any]) -> bool:
    """An assistant message is an ACTION if it carries a structured tool call."""
    return bool(msg.get("tool_calls"))


def _plan_text(messages: list[dict[str, Any]]) -> str:
    """All assistant-authored text where a plan can live: the native reasoning channel
    (where <goal>/<plan> ride for plan-mode rows) plus visible content."""
    return "\n".join((m.get("reasoning") or "") + "\n" + (m.get("content") or "")
                     for m in messages if m.get("role") == "assistant")


def _skill_loads(messages: list[dict[str, Any]]) -> list[str]:
    """Skill names loaded via load_skill{name:NAME}, in order."""
    return [args.get("name", "") for msg in messages if msg.get("role") == "assistant"
            for name, args in _msg_calls(msg) if name == LOAD_SKILL]


def _actions(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Ordered (kind, name) stream of harness actions: ('skill'|'tool', name).
    A load_skill call is a 'skill' action (name = the loaded skill)."""
    out: list[tuple[str, str]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for name, args in _msg_calls(msg):
            if name == LOAD_SKILL:
                out.append(("skill", args.get("name", "")))
            else:
                out.append(("tool", name))
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
    violations.extend(_plan_structure(row))
    violations.extend(_audit_boxes(row, calls))
    violations.extend(_followup_grounded(row))
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


def _tool_calls(messages: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], str]]:
    calls = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for name, args in _msg_calls(message):
            calls.append((name, args, _canonical(name, args)))
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
    for name in loaded:                       # every loaded skill must be a listed skill
        if name not in skills:
            out.append(Violation("selected_skill_exists", f"unknown skill loaded: {name}"))
    for selected in row["selected_skills"]:
        if selected not in loaded:
            out.append(Violation("skill_loaded_after_selection", selected))
    return out


def _final(messages: list[dict[str, Any]]) -> list[Violation]:
    finals = [m["content"] for m in messages if m.get("role") == "assistant" and not _is_action(m)]
    if not finals:
        return [Violation("final_no_xml", "missing final assistant answer")]
    final = finals[-1]
    # No raw action markup leaks into the user-facing answer — neither the old
    # custom XML nor the native control tokens.
    leaks = any(t in final for t in ("<tool>", "</tool>", "<skill>", "</skill>",
                                     "<|tool_call>", "<|channel>", "<|tool_response>"))
    return [Violation("final_no_xml", "final contains raw action markup")] if leaks else []


def _tool_names(calls: list[tuple[str, dict[str, str], str]], tools: dict[str, Any]) -> list[Violation]:
    return [Violation("known_tool_only", name) for name, _, _ in calls if name not in tools]


def _tool_args(calls: list[tuple[str, dict[str, str], str]], tools: dict[str, Any]) -> list[Violation]:
    out: list[Violation] = []
    for name, args, _ in calls:
        schema = tools.get(name, {}).get("args", {})
        for arg, rule in schema.items():
            if rule == "required" and arg not in args:
                out.append(Violation("args_match_schema", f"{name}.{arg} required"))
            # enum compared as strings: structured args are typed (top:3 int) but
            # the schema lists string choices (["1".."5"]).
            if isinstance(rule, list) and arg in args and str(args[arg]) not in [str(r) for r in rule]:
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
              if m.get("role") == "assistant" and not _is_action(m)]
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
        # A board read showing a real last move proves a game is in progress (the
        # post-game review context that licenses has_history tools).
        if text.startswith("board_state:") and "last_move=" in text and "last_move=none" not in text:
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
        for name, args in _msg_calls(message):
            if name != "move":
                continue
            san = str(args.get("san", ""))
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


def _one_tool_per_message(messages: list[dict[str, Any]]) -> list[Violation]:
    """One tool call per inference step: each assistant message holds at most one
    structured tool call. Many calls across the loop are fine — they live in
    separate assistant messages."""
    out: list[Violation] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        if len(_msg_calls(message)) > 1:
            out.append(Violation("one_tool_per_message", "multiple actions in one inference step"))
    return out


def _reasoning_mode(row: dict[str, Any]) -> list[Violation]:
    """Dual-mode integrity: a `fast` row must carry NO reasoning channel (native thinking
    rides the `reasoning` field; fast = answer directly). think/auto carry none in training
    (native at serve); plan carries <goal>/<plan> there. This keeps fast-vs-think a real
    toggle rather than an always-on reflex."""
    mode = (row.get("reasoning_mode") or "").strip().lower()
    if mode != "fast":
        return []
    for m in row.get("messages", []):
        if m.get("role") == "assistant" and (m.get("reasoning") or "<think>" in (m.get("content") or "")):
            return [Violation("reasoning_mode_fast_no_think", "fast row carries reasoning")]
    return []


def _plan_structure(row: dict[str, Any]) -> list[Violation]:
    """PLAN-mode (Stage 1/2) deterministic gates:
    - goal_before_plan: a <goal> AND a <plan> exist, and <goal> comes first (commit the
      objective before the checklist — the anti-early-stop contract).
    - plan_boxes_bound: every checkbox's binding "(name)" maps to a real listed skill or
      tool (or the literal synthesis marker), so the serve box-tracking gate can map each
      box to an executed action — no dangling boxes."""
    rules = row.get("acceptance_rules", [])
    if "goal_before_plan" not in rules and "plan_boxes_bound" not in rules:
        return []
    text = _plan_text(row["messages"])
    out: list[Violation] = []
    gm, pm = _GOAL_TAG.search(text), _PLAN_TAG.search(text)
    if "goal_before_plan" in rules:
        if not gm or not pm:
            out.append(Violation("goal_before_plan", "missing <goal> or <plan>"))
        elif gm.start() > pm.start():
            out.append(Violation("goal_before_plan", "<plan> appears before <goal>"))
    if "plan_boxes_bound" in rules and pm:
        names = {s.get("name") for s in row.get("skills_index", [])}
        names |= {t.get("name") for t in row.get("tool_manifest", [])}
        for binding in _BOX_BIND.findall(pm.group(1)):
            b = binding.strip()
            if b in ("none", "synthesize", "synthesis"):
                continue
            if b not in names:
                out.append(Violation("plan_boxes_bound", f"box binding '{b}' not a listed skill/tool"))
    return out


def _audit_boxes(row: dict[str, Any], calls: list[tuple[str, dict[str, str], str]]) -> list[Violation]:
    """Stage 2: every plan box bound to `python` must be CLOSED by a real python audit —
    the model runs the script and reads the output, it does not assert. Enforced only when
    the audit procedure actually ran (plan-audit in selected_skills); an honest-partial
    abort (audit skill disabled -> not selected) is exempt, since the point is it couldn't."""
    if "audit_boxes_grounded" not in row.get("acceptance_rules", []):
        return []
    if "plan-audit" not in row.get("selected_skills", []):
        return []                                   # honest-partial abort -> nothing to audit
    text = _plan_text(row["messages"])
    pm = _PLAN_TAG.search(text)
    if not pm:
        return [Violation("audit_boxes_grounded", "missing <plan>")]
    py_boxes = sum(1 for b in _BOX_BIND.findall(pm.group(1)) if b.strip() == "python")
    py_calls = sum(1 for name, _, _ in calls if name == "python")
    if py_calls < py_boxes:
        return [Violation("audit_boxes_grounded", f"{py_calls} python audits for {py_boxes} checkable boxes")]
    return []


def _followup_grounded(row: dict[str, Any]) -> list[Violation]:
    """Anti-confabulation across turns: a FOLLOW-UP turn whose final ASSERTS a position
    fact (a standing word, a pawn number, or a SAN move) must RE-GROUND it with a tool
    call in that same turn — the model must never answer "same read as before" from
    memory. The trained turn is everything after the last context-only (`train: False`)
    assistant message; single-turn rows have no such message and are exempt (their whole
    turn is grounded by the other rules). Unconditional so a new archetype can't bypass it."""
    messages = row.get("messages", [])
    last_ctx = max((i for i, m in enumerate(messages)
                    if m.get("role") == "assistant" and m.get("train") is False), default=-1)
    if last_ctx < 0:
        return []                                   # single-turn row -> exempt
    segment = messages[last_ctx + 1:]
    finals = [m.get("content", "") for m in segment
              if m.get("role") == "assistant" and not _is_action(m)]
    if not finals:
        return []
    final = finals[-1]
    if not (_STANDING.search(final) or _FACT.search(final)):
        return []                                   # asserts no fact -> a clarify/ack is fine
    grounded = any(name != LOAD_SKILL for m in segment for name, _ in _msg_calls(m))
    if not grounded:
        return [Violation("followup_grounded",
                          "follow-up asserts a position fact with no tool call in the turn")]
    return []


def _injection(row: dict[str, Any]) -> list[Violation]:
    if "tool_text_is_data" not in row["acceptance_rules"]:
        return []
    finals = [m.get("content", "").lower() for m in row["messages"] if m.get("role") == "assistant"]
    final = finals[-1] if finals else ""
    bad = ("ignore previous" in final) or ("system overridden" in final)
    return [Violation("tool_text_is_data", "followed injected text")] if bad else []
