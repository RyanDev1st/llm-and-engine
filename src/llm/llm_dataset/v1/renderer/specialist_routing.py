"""V1_U specialist routing: from the flat chess catalog, pick the RIGHT specialist by
intent/context and use it — game-reviewer for "how did I play", opening-advisor for "what
opening is this", tactical-puzzles for "give me a puzzle" — then ground the answer in its
tool result.

Deliberately MODERATE: the model already routes skills well, so this just hardens
intent -> specialist mapping (and the chess-coach core slices A-K already cover the
generalist). It is NOT a slang/phrasing memorizer — prompts vary by style, but the lesson
is reading intent and picking the listed skill whose description fits."""
from __future__ import annotations

import random
from typing import Any

from ..catalog import chess_skills, chess_tools
from ..specialists import SPECIALISTS, Specialist, scene
from .leadins import lead
from .thinking import gated_answer, gated_think, pick_mode, prepend_open_goal

_GOALS = {
    "game-reviewer": "review how they played",
    "opening-advisor": "identify the opening and its plans",
    "tactical-puzzles": "set a tactical puzzle",
}
_LAST_MOVES = ("Nf6", "Bd6", "Qc7", "Nc6", "exd5", "O-O")
_STYLES = ("formal", "casual", "slang", "typo", "anxious", "beginner")


def _join(*parts: str) -> str:
    return "\n".join(p for p in parts if p)


def _style_prompt(base: str, seed: int) -> str:
    style = _STYLES[seed % len(_STYLES)]
    if style == "formal":
        return f"Please {base}"
    if style == "slang":
        return f"yo, {base}"
    if style == "typo":
        return f"{base} pls"
    if style == "anxious":
        return f"I'm worried here - {base}"
    if style == "beginner":
        return f"I'm new to chess; {base}"
    return base


def _ground_board(seed: int, goal: str, mode: str) -> list[dict]:
    """A board read establishing a game-in-progress, so the has_history specialist tools
    (accuracy_report / name_opening) are licensed (validate._has_move_history reads it)."""
    lm = random.Random(seed * 47 + 9).choice(_LAST_MOVES)
    call = _join(gated_think(seed, "board_state", 0, mode=mode, kind="routine", goal=goal, have=""),
                 lead(seed, "board_state", 0), "<tool>board_state fields=all</tool>")
    result = f"board_state: turn=white, last_move={lm}, check=no, legal_count=31"
    return [{"role": "assistant", "content": call}, {"role": "tool", "content": result}]


def render_specialist_routing_row(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    spec: Specialist = SPECIALISTS[seed % len(SPECIALISTS)]
    mode = pick_mode(seed)
    goal = _GOALS[spec.skill]
    messages: list[dict[str, str]] = [{"role": "user", "content": _style_prompt(rng.choice(spec.prompts), seed)}]

    if spec.applies_when == "has_history":           # review/opening: read the game first
        messages += _ground_board(seed, goal, mode)

    messages.append({"role": "assistant", "content": _join(
        gated_think(seed, "load_skill", 1, mode=mode, kind="select", goal=goal),
        lead(seed, "load_skill", 1), f"<skill>{spec.skill}</skill>")})
    messages.append({"role": "tool", "content": spec.body()})

    call, tool_result, finding = scene(spec, seed)
    tool_call = f"<tool>{spec.tool}{(' ' + call) if call else ''}</tool>"
    messages.append({"role": "assistant", "content": _join(
        gated_think(seed, spec.tool, 2, mode=mode, kind="execute", goal=goal, have="skill"),
        lead(seed, spec.tool, 2), tool_call)})
    messages.append({"role": "tool", "content": tool_result})

    ans = gated_answer(seed, goal, mode=mode)
    final = finding[0].upper() + finding[1:] + "."
    messages.append({"role": "assistant", "content": f"{ans}\n{final}" if ans else final})
    prepend_open_goal(messages, seed, mode, goal)
    return _envelope(seed, messages, spec, mode)


import re as _re
_TOOL = _re.compile(r"<tool>\s*([a-z_][a-z0-9_]*)")


def _envelope(seed: int, messages: list[dict], spec: Specialist, mode: str) -> dict[str, Any]:
    expected = [m for c in (msg["content"] for msg in messages if msg["role"] == "assistant")
                for m in _TOOL.findall(c)]
    return {
        "id": f"v1_u_route_{seed:09d}",
        "slice": "V1_U_specialist_routing",
        "kind": "specialist_routing",
        "reasoning_mode": mode,
        "intent": f"v1_u_{seed:06d}",
        "plugin_context": {},
        "skills_index": chess_skills(),
        "selected_skills": [spec.skill],
        "tool_manifest": chess_tools(),
        "expected_tool_calls": expected,
        "grounding_sources": [],
        "messages": messages,
        "acceptance_rules": ["final_no_xml", "known_tool_only", "args_match_schema",
                             "selected_skill_exists", "skill_body_strict"],
        "position_fen": None,
        "stockfish_truth": None,
    }
