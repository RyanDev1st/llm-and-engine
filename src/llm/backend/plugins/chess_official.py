"""The chess-official plugin: owns the default/official tools + skills, and a
prompt-start hook that injects this plugin's RUNTIME STATE (the live board) into the
system prompt each turn — so the model doesn't spend a board_state round-trip to read
a board it can't see.

It deliberately does NOT pre-load any skill body: skills stay progressive-disclosure
(name + description in the catalog; the model decides to `load_skill` the body it
needs). That keeps the harness plug-and-play and domain-agnostic — a different plugin
registers its own state hook and its own skills the same way, no hard-coded bodies.
"""
from __future__ import annotations

import chess

from llm_dataset.v1.catalog import official_tools
from ..skills import load_skills

NAME = "chess-official"


def tools() -> list[dict]:
    """The default tool manifest this plugin provides."""
    return official_tools()


def skills() -> list:
    """The skills this plugin bundles (name + description only; bodies load on demand)."""
    return load_skills()


def _board_line(game) -> str:
    b = game.board
    turn = "white" if b.turn == chess.WHITE else "black"
    last = game.san_stack[-1] if game.san_stack else "none"
    return (f"turn={turn}, last_move={last}, check={'yes' if b.is_check() else 'no'}, "
            f"legal_moves={b.legal_moves.count()}, game_over={game.over_status() or 'no'}, "
            f"fen={b.fen()}")


def prompt_start(context: dict) -> str:
    """Prompt-start hook (Anthropic-style additional-context injection): inject this
    plugin's live runtime state — the current board — so the model need not call
    board_state to read it. NOT a skill-body preload; skills load on demand.
    `context` may carry {"game": <Game>}."""
    game = context.get("game")
    if game is None:
        return ""
    return ("LIVE BOARD (current snapshot from the chess-official plugin; no need to "
            "call board_state to read it): " + _board_line(game))
