---
name: isolated-pawn-play
description: Use when the user asks how to play with or against an isolated queen pawn.
---

# isolated-pawn-play
When to use: the user asks about an isolated pawn position.
Steps:
1. Call board_state with fields=fen to confirm the isolani.
2. Decide whether to blockade it or use its space.
3. Give the plan for the side you are coaching.
Constraint: the isolani's owner attacks; the other side blockades.
