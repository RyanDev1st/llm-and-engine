---
name: endgame-technique
description: Use when the position is an endgame and the user wants to convert a win or hold a draw.
---

# endgame-technique
When to use: few pieces remain and the user asks how to finish or hold the game.
Steps:
1. Call board_state to confirm material and whose move it is.
2. Use best_move with series to read the winning or holding line.
3. Name the technique and its key idea.
Constraint: verify the line with the engine before promising a result.
