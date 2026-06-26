"""puzzles plugin: a version-controlled tactical-puzzle COACH skill.

It owns no new tools — it uses the official `random_position` (local curated bank) and
`fetch_puzzle` (real rated Lichess puzzle). Its whole value is the SKILL BODY: it guides
the model through the full coaching loop — set a puzzle, present it, give a hint when
asked, and REVEAL the solution when the user is stuck — instead of mindlessly
re-generating a new puzzle on every reply. The earlier user-authored skill only said
"generate puzzles", so the model (which now faithfully follows skill bodies) kept
re-rolling and never helped a stuck solver. This is the fix: better guidance, in the
skill, not more serve-side determinism."""
from __future__ import annotations

NAME = "puzzles"
TOOLS: list[dict] = []   # uses the official random_position + fetch_puzzle

_BODY = """---
name: tactical-puzzles
description: Set and coach tactical puzzles.
---

# tactical-puzzles

Coach ONE puzzle at a time. The position tools set the board and return the FEN/motif, but they
do NOT reveal the solution. To check or reveal the answer, call `best_move` on the current board;
that result is GROUND TRUTH. Never invent a move.

To START a puzzle, act THIS turn — never ask the user which type or "specific or random?". Just
call `random_position kind=puzzle` (the default). Only use `fetch_puzzle` instead if they
explicitly asked for a real / rated / Lichess puzzle. The result gives the FEN, whose move it is,
and the motif. Then present it in this order: state whose turn it is, name only the motif to hunt
for (e.g. "look for a fork"), and ask them to find the move. Then stop and wait.

When they REPLY, read intent — do not auto-set a new puzzle:
- a move → call `best_move` if you have not already grounded the answer; compare the user's move
  to that result. A match = praise + one line on why it works; a miss = say so, then give a hint
  toward the best move (the piece or the idea), and let them retry.
- stuck or asking for it ("idk", "I'm bad", "help", "hint", "show me", "what's the answer",
  "give up") → reveal now: call `best_move`, state that move, and explain in one or two lines
  why it wins (the tactic — fork/pin/skewer/mate). Stay on this puzzle.
- "another / harder / different" → only THEN call a position tool again.

A stuck user wants the answer and the idea, not a fresh position. Keep it warm and short; you
are coaching, not quizzing."""

SKILLS = [{
    "name": "tactical-puzzles",
    "description": ("Use when the user wants a tactical puzzle, to practice or hone tactics, "
                    "or to be coached through a combination."),
    "body": _BODY,
}]
