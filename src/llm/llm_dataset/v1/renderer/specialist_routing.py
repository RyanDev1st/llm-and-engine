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
from .tags import scene_args, skill_call_msg, tool_call_msg, tool_calls_of, tool_result_msg
from .thinking import pick_mode

_LAST_MOVES = ("Nf6", "Bd6", "Qc7", "Nc6", "exd5", "O-O")
_STYLES = ("formal", "casual", "slang", "typo", "anxious", "beginner")


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


def _ground_board(seed: int) -> list[dict]:
    """A board read establishing a game-in-progress, so the has_history specialist tools
    (accuracy_report / name_opening) are licensed (validate._has_move_history reads it)."""
    lm = random.Random(seed * 47 + 9).choice(_LAST_MOVES)
    result = f"board_state: turn=white, last_move={lm}, check=no, legal_count=31"
    return [tool_call_msg("board_state", {"fields": "all"}),
            tool_result_msg("board_state", result)]


def render_specialist_routing_row(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    spec: Specialist = SPECIALISTS[seed % len(SPECIALISTS)]
    mode = pick_mode(seed)
    messages: list[dict[str, Any]] = [{"role": "user", "content": _style_prompt(rng.choice(spec.prompts), seed)}]

    if spec.applies_when == "has_history":           # review/opening: read the game first
        messages += _ground_board(seed)

    messages.append(skill_call_msg(spec.skill))
    messages.append(tool_result_msg("load_skill", spec.body()))

    call, tool_result, finding = scene(spec, seed)
    messages.append(tool_call_msg(spec.tool, scene_args(call)))
    messages.append(tool_result_msg(spec.tool, tool_result))

    final = finding[0].upper() + finding[1:] + "."
    messages.append({"role": "assistant", "content": final})
    return _envelope(seed, messages, spec, mode)


def _envelope(seed: int, messages: list[dict], spec: Specialist, mode: str) -> dict[str, Any]:
    expected = [tc["name"] for msg in messages if msg["role"] == "assistant"
                for tc in tool_calls_of(msg) if tc["name"] != "load_skill"]
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
