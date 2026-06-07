---
name: candidate-moves
description: Use when the user wants a shortlist of the best moves to consider.
---

# candidate-moves
When to use: the user asks for candidate moves or the top options.
Steps:
1. Call best_move with top=3 to list the leading moves.
2. Order them by the engine score.
3. Add a one-line idea for the top choice.
Constraint: list only engine-supported candidates with their scores.
