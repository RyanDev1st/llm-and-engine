"""Stage 1 — V1_S_compound_plan: goal-driven completion across TWO chess specialists.

Teaches the anti-early-stop loop: a request that needs two specialists to fully answer
("review my game AND tell me what opening to study"). The model commits BOTH goals, lists
the steps as a <plan>, then DOES EVERY box (load skill -> call its tool -> read result)
before synthesizing across both findings — instead of doing one and half-answering.

v5 pure-chess + flat catalog: skills/tools come from the flat chess catalog (no plugin
gating), the boxes bind to the two chosen specialists. has_history specialist tools are
licensed by the post-game context (the user has played a game), so applies_when_respected
is not enforced on these rows. Honest-partial is dropped here (its disabled-skill trigger
doesn't exist in a flat catalog); split-determinism + honest reporting live in audited-plan."""
from __future__ import annotations

import random
from typing import Any

from ..catalog import chess_skills, chess_tools
from ..specialists import Specialist, pick_two, scene
from .planning import goal_block, plan_block
from .thinking import gated_think

_LEAD_LOAD = ("Loading the skill for this part.", "Next skill for the next box.",
              "Pulling the skill this box needs.")
_LEAD_TOOL = ("Now its data.", "Running its tool.", "Getting the specifics.")
# A review/opening plan is about a game already played; the coach reads the board first so
# the has_history specialist tools (accuracy_report, name_opening) are licensed by a real
# game in progress (validate._has_move_history reads this last_move).
_LAST_MOVES = ("Nf6", "Bd6", "Qc7", "Rfe8", "Nc6", "Bb4", "exd5", "cxd4")


def _ground_board(seed: int, goal: str, mode: str) -> list[dict]:
    lm = random.Random(seed * 47 + 9).choice(_LAST_MOVES)
    call = _join(gated_think(seed, "board_state", 0, mode=mode, kind="routine", goal=goal, have=""),
                 "Reading the game first.", "<tool>board_state fields=all</tool>")
    result = f"board_state: turn=white, last_move={lm}, check=no, legal_count=31"
    return [{"role": "assistant", "content": call}, {"role": "tool", "content": result}]


def _pick(seed: int, step: int, pool: tuple[str, ...]) -> str:
    return random.Random(seed * 31 + step).choice(pool)


def _join(*parts: str) -> str:
    return "\n".join(p for p in parts if p)


def _compound_prompt(r: random.Random, a: Specialist, b: Specialist) -> str:
    pa, pb = r.choice(a.prompts), r.choice(b.prompts)
    return r.choice((f"{pa} and also {pb}",
                     f"two things: {pa.rstrip('?')}; then {pb}",
                     f"{pa} — and {pb} while you're at it"))


def _box_steps(seed: int, spec: Specialist, step0: int, goal: str, mode: str) -> tuple[list[dict], str]:
    """One box = load skill -> call its tool. Returns (messages, finding)."""
    call, tool_result, finding = scene(spec, seed + step0)
    tool_call = f"<tool>{spec.tool}{(' ' + call) if call else ''}</tool>"
    msgs = [
        {"role": "assistant", "content": _join(
            gated_think(seed, "load_skill", step0, mode=mode, kind="select", goal=goal),
            _pick(seed, step0, _LEAD_LOAD), f"<skill>{spec.skill}</skill>")},
        {"role": "tool", "content": spec.body()},
        {"role": "assistant", "content": _join(
            gated_think(seed, spec.tool, step0 + 1, mode=mode, kind="execute", goal=goal, have="skill"),
            _pick(seed, step0 + 1, _LEAD_TOOL), tool_call)},
        {"role": "tool", "content": tool_result},
    ]
    return msgs, finding


def render_compound_plan_row(seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    a, b = pick_two(seed)
    mode = "plan"
    # Commit BOTH goals (compound request = two distinct asks), then plan covers each.
    goal_a = f"the {a.skill.replace('-', ' ')} ask"
    goal_b = f"the {b.skill.replace('-', ' ')} ask"
    goal_text = f"{goal_a} and {goal_b}"
    boxes = [(f"handle the {a.skill.replace('-', ' ')} part", a.skill),
             (f"handle the {b.skill.replace('-', ' ')} part", b.skill),
             ("synthesize one combined answer", "none")]

    messages: list[dict[str, str]] = [{"role": "user", "content": _compound_prompt(rng, a, b)}]
    messages.append({"role": "assistant",
                     "content": goal_block(seed, [goal_a, goal_b]) + "\n" + plan_block(boxes)})
    messages += _ground_board(seed, goal_text, mode)   # establish the game-in-progress first
    box_a, finding_a = _box_steps(seed, a, 1, goal_text, mode)
    messages += box_a
    box_b, finding_b = _box_steps(seed, b, 3, goal_text, mode)
    messages += box_b
    messages.append({"role": "assistant",
                     "content": f"On the first: {finding_a}. On the second: {finding_b}."})
    return _envelope(seed, messages, [a.skill, b.skill], mode)


import re as _re
_TOOL = _re.compile(r"<tool>\s*([a-z_][a-z0-9_]*)")


def _envelope(seed: int, messages: list[dict], selected: list[str], mode: str) -> dict[str, Any]:
    expected = [m for c in (msg["content"] for msg in messages if msg["role"] == "assistant")
                for m in _TOOL.findall(c)]
    return {
        "id": f"v1_s_compound_{seed:09d}",
        "slice": "V1_S_compound_plan",
        "kind": "compound_plan",
        "reasoning_mode": mode,
        "intent": f"v1_s_{seed:06d}",
        "plugin_context": {},
        "skills_index": chess_skills(),
        "selected_skills": selected,
        "tool_manifest": chess_tools(),
        "expected_tool_calls": expected,
        "grounding_sources": [],
        "messages": messages,
        "acceptance_rules": ["final_no_xml", "known_tool_only", "args_match_schema",
                             "selected_skill_exists", "skill_body_strict",
                             "goal_before_plan", "plan_boxes_bound"],
        "position_fen": None,
        "stockfish_truth": None,
    }
