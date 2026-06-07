---
name: piece-activity
description: Use when the user wants to improve a passive piece or find their worst-placed piece.
---

# piece-activity
When to use: the user asks which piece to improve or how to activate one.
Steps:
1. Call list_pieces to see where each piece stands.
2. Identify the least active piece and a better square.
3. Give the move that reroutes it.
Constraint: improve the worst piece first; keep the move legal.
