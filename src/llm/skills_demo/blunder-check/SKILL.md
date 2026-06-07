---
name: blunder-check
description: Use when the user wants to know if their last move was a blunder or mistake.
---

# blunder-check
When to use: the user asks whether the move they just played was bad.
Steps:
1. Call review_move to grade the last move against the best.
2. Report the label and the centipawn delta.
3. If it was a blunder, name the move that refutes it.
Constraint: do not guess the grade; use the review_move output.
