---
name: stalemate-awareness
description: Use when the user is winning and risks stalemating the enemy king.
---

# stalemate-awareness
When to use: the user is up big material and asks how to finish safely.
Steps:
1. Call board_state to check the opponent's legal_count.
2. If the enemy is nearly stuck, leave it a legal move.
3. Give a mating path that avoids stalemate.
Constraint: never give a move that stalemates while you are winning.
