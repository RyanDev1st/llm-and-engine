---
name: board-recall
description: Use when the user asks what the current board looks like, the FEN, or move history.
---

# board-recall
When to use: the user asks to recall the board state, position, or history.
Steps:
1. Call board_state with fields=all for turn, FEN, last move, check, history.
2. Relay the facts the user asked for.
3. Offer to evaluate or list legal moves next.
Constraint: report board facts only from board_state; never recite from memory.
