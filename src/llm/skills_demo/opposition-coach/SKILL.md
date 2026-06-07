---
name: opposition-coach
description: Use when the user asks about king opposition in a pawn endgame.
---

# opposition-coach
When to use: the user asks about the opposition or king-and-pawn technique.
Steps:
1. Call board_state to read the king and pawn squares.
2. Use best_move to find the move that takes the opposition.
3. Explain who holds the opposition and why it matters.
Constraint: the king move must follow the engine's drawing or winning line.
