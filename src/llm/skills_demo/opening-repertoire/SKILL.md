---
name: opening-repertoire
description: Use when the user asks what to open with, wants a repertoire, or how to meet a first move.
---

# opening-repertoire
When to use: the user asks what to play in the opening or wants a repertoire.
Steps:
1. Call board_state to read the moves played so far.
2. Ask ask_chessbot for a principled line that fits the position.
3. Name the one middlegame plan the opening aims for.
Constraint: suggest only legal, principled moves; do not invent forced theory.
