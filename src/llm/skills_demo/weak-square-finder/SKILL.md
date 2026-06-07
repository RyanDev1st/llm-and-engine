---
name: weak-square-finder
description: Use when the user asks about weak squares, holes, or outpost targets.
---

# weak-square-finder
When to use: the user asks where the weak squares or holes are.
Steps:
1. Call board_state with fields=fen to map the pawns.
2. Find squares no enemy pawn can defend.
3. Name the piece that should occupy the best hole.
Constraint: a hole must be a square no pawn can ever cover.
