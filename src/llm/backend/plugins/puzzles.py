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

Coach ONE puzzle at a time. The position tools return the solution as `answer=<SAN>` (plus a
score) — that is GROUND TRUTH; use it, never invent a move.

To START a puzzle, call ONE position tool this turn: `random_position kind=puzzle` (a curated
tactic) or `fetch_puzzle` (a real rated Lichess puzzle). The result gives the FEN, whose move
it is, and `answer`. Then present it in this order: state whose turn it is, name only the motif
to hunt for (e.g. "look for a fork"), and ask them to find the move — keep the move itself for
later. Then stop and wait.

When they REPLY, read intent — do not auto-set a new puzzle:
- a move → compare to `answer`: a match = praise + one line on why it works; a miss = say so,
  then a hint toward `answer` (the piece or the idea), and let them retry.
- stuck or asking for it ("idk", "I'm bad", "help", "hint", "show me", "what's the answer",
  "give up") → reveal now: state the move (`answer`) and explain in one or two lines why it
  wins (the tactic — fork/pin/skewer/mate). You may call `best_move` to ground it. Stay on
  this puzzle.
- "another / harder / different" → only THEN call a position tool again.

A stuck user wants the answer and the idea, not a fresh position. Keep it warm and short; you
are coaching, not quizzing."""

SKILLS = [{
    "name": "tactical-puzzles",
    "description": ("Use when the user wants a tactical puzzle, to practice or hone tactics, "
                    "or to be coached through a combination."),
    "body": _BODY,
}]
