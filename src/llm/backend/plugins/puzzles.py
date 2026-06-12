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

Coach the user through tactical puzzles. The position tools return the SOLUTION in their
result as `answer=<SAN>` (and a score) — that answer is GROUND TRUTH; use it, never invent one.

How to run a puzzle session:
1. To start a puzzle, call ONE position tool: `random_position kind=puzzle` (a curated
   tactic) or `fetch_puzzle` (a real rated Lichess puzzle). Read the result: it has the
   FEN, whose move it is, and `answer=<SAN>`.
2. Present the puzzle to the user: state whose turn it is and ask them to find the move.
   Do NOT reveal `answer` yet — let them try.
3. Then STOP and wait for the user. One puzzle at a time.

When the user replies, read what they want — do NOT just set another puzzle:
- They give a move → say if it matches `answer`. If right, praise + one line on why it
  works. If wrong, say so and give a hint toward `answer` (the piece or the idea), let
  them try again.
- They are STUCK or ask for help — "I don't know", "I'm bad", "help", "hint", "show me",
  "what's the answer", "give up" → REVEAL the solution: state the best move (`answer`) and
  explain in one or two lines WHY it wins (the tactic — fork/pin/skewer/mate). You may call
  `best_move` or `board_state` to ground the explanation. Do NOT generate a new puzzle.
- They ask for ANOTHER / a harder / different puzzle → only THEN call a position tool again.

Never re-roll a puzzle just because the user is unsure. A stuck user wants the answer and
the idea, not a fresh position. Keep it warm and short; you are coaching, not quizzing."""

SKILLS = [{
    "name": "tactical-puzzles",
    "description": ("Use when the user wants tactical puzzles or to practice/hone tactics. "
                    "Sets a puzzle, then COACHES them through it — hints, checks their move, "
                    "and reveals the solution when they are stuck. Does not re-generate a "
                    "puzzle unless the user asks for another."),
    "body": _BODY,
}]
