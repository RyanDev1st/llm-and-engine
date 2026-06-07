---
name: game-review
description: Use when the user wants a full post-mortem of a game just played.
---

# game-review
When to use: the user asks to review the whole game or find where it went wrong.
Steps:
1. Call review_move to grade the most recent move.
2. Identify the moves where the evaluation swung most.
3. Summarize the turning point and one lesson.
Constraint: grade moves from review_move output, not from memory.
