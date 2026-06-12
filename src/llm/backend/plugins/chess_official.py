"""The chess-official plugin: owns the default/official tools + skills, and a
prompt-start hook that pre-loads its always-on context — the chess-coach skill body
and a live board snapshot — into the system prompt. With both already present the
model skips the load_skill + board_state round-trips it was trained to make, which is
the largest serve-side latency win (each is a whole extra model generation).
"""
from __future__ import annotations

import chess

from llm_dataset.v1.catalog import official_tools
from ..skills import load_skills

NAME = "chess-official"
ALWAYS_ON_SKILL = "chess-coach"  # the default skill, pre-loaded by the hook


def tools() -> list[dict]:
    """The default tool manifest this plugin provides."""
    return official_tools()


def skills() -> list:
    """The skills this plugin bundles (catalog entries with bodies)."""
    return load_skills()


def _skill_body(content: str) -> str:
    """Strip YAML frontmatter, return the markdown body."""
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            return content[end + 4:].lstrip()
    return content.strip()


def _board_line(game) -> str:
    b = game.board
    turn = "white" if b.turn == chess.WHITE else "black"
    last = game.san_stack[-1] if game.san_stack else "none"
    return (f"turn={turn}, last_move={last}, check={'yes' if b.is_check() else 'no'}, "
            f"legal_moves={b.legal_moves.count()}, game_over={game.over_status() or 'no'}, "
            f"fen={b.fen()}")


def prompt_start(context: dict) -> str:
    """Prompt-start hook: inject the always-on coach skill body + the live board so the
    model has them up front. `context` may carry {"game": <Game>}."""
    parts: list[str] = []
    body = next((s.content for s in load_skills() if s.name == ALWAYS_ON_SKILL), "")
    if body:
        parts.append("ACTIVE SKILL — chess-coach (pre-loaded; follow it, do NOT call "
                     "load_skill for it):\n" + _skill_body(body))
    game = context.get("game")
    if game is not None:
        parts.append("LIVE BOARD (current snapshot; do NOT call board_state to read it): "
                     + _board_line(game))
    return "\n\n".join(parts)
