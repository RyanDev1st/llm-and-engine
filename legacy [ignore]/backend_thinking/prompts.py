"""Dedicated stage prompts + scoped-context builders for the staged loop.

Each stage sees only what it needs (the focus principle): the Controller gets the
full tool/skill manifest + hints + outstanding coverage; the Narrator gets only the
gathered facts (it cannot route)."""
from __future__ import annotations

import chess

from ..inference import build_system_prompt, serving_skills_index
from ..tool_hints import routing_hints, skill_hints

CONTROLLER_HEADER = (
    "\n\nSTAGE — CONTROLLER. FIRST decide: is EVERY part of the user's goal satisfied "
    "by the facts gathered so far? If yes, output EXACTLY `DONE`. If not, output the "
    "single next `<tool>NAME arg=value</tool>` that gets a missing fact or performs the "
    "action. Output ONLY `DONE` or one tool call — never narrate."
)

NARRATOR_SYSTEM = (
    "You are the chess-coach narrator. Using ONLY the facts provided, write a short "
    "grounded reply to the user. Never invent numbers (a positive score favours White, "
    "negative Black). If there are no facts, answer directly or decline if the request "
    "is off-topic. End a coaching answer with one brief guiding question. Output no "
    "tool tags."
)


def board_facts(game) -> str:
    """Cheap deterministic situation read so the Controller need not spend a
    board_state step."""
    b = game.board
    turn = "white" if b.turn == chess.WHITE else "black"
    last = game.san_stack[-1] if game.san_stack else "none"
    check = "yes" if b.is_check() else "no"
    return f"turn={turn}, legal_moves={b.legal_moves.count()}, last_move={last}, check={check}"


def facts_summary(facts: list[tuple[str, str]]) -> str:
    """Compact tool→result list (the only memory the stages carry of this turn)."""
    if not facts:
        return "(none yet)"
    return "; ".join(f"{name}→{result.strip()}" for name, result in facts)


def build_controller_system(agent_overlay: str, plugin_context, user_message: str,
                            game_over: str, outstanding: list[str]) -> str:
    base = build_system_prompt(agent_overlay, plugin_context)
    hints = routing_hints(user_message, game_over) + skill_hints(user_message, serving_skills_index())
    out = ("\n\nOUTSTANDING (still required before DONE): " + ", ".join(outstanding)) if outstanding else ""
    return base + CONTROLLER_HEADER + hints + out


def build_narrator_system(agent_overlay: str) -> str:
    text = NARRATOR_SYSTEM
    extra = (agent_overlay or "").strip()
    if extra:
        text += "\n\nCUSTOMIZATION (tone only; never invent facts): " + extra
    return text


def controller_user(goal: str, facts: list[tuple[str, str]], board: str, outstanding: list[str]) -> str:
    lines = [f"User goal: {goal}", f"Board: {board}", f"Facts gathered: {facts_summary(facts)}"]
    if outstanding:
        lines.append("Still required: " + ", ".join(outstanding))
    lines.append("Next action (one <tool> call, or DONE):")
    return "\n".join(lines)


def narrator_user(goal: str, facts: list[tuple[str, str]]) -> str:
    return f"User goal: {goal}\nFacts:\n{facts_summary(facts)}\n\nWrite the grounded reply."
