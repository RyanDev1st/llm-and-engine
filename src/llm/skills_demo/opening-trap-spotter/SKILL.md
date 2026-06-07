---
name: opening-trap-spotter
description: Use when the user worries about an opening trap or gambit they might fall into.
---

# opening-trap-spotter
When to use: the user asks if a line is a trap or how to avoid one.
Steps:
1. Call board_state to read the exact move order.
2. Ask ask_chessbot whether a known trap lurks here.
3. Give the safe move that sidesteps it.
Constraint: do not enter a line you cannot refute over the board.
