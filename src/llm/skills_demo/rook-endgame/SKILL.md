---
name: rook-endgame
description: Use when the position is a rook endgame needing technique like Lucena or Philidor.
---

# rook-endgame
When to use: a rook endgame arises and the user asks how to win or draw it.
Steps:
1. Call board_state to confirm pawns, rooks, and king places.
2. Use best_move series to find the technical line.
3. Name the method (Lucena, Philidor, active rook).
Constraint: verify the drawing or winning method with the engine line.
