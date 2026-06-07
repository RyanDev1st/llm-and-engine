---
name: pawn-structure
description: Use when the user asks about pawn chains, islands, doubled, backward, or passed pawns.
---

# pawn-structure
When to use: the user asks about the pawn structure or a pawn weakness.
Steps:
1. Call board_state with fields=fen to read the pawn skeleton.
2. Ask ask_chessbot to name the structure and its plans.
3. Point out the one pawn that decides the plan.
Constraint: describe only pawns present in the position.
