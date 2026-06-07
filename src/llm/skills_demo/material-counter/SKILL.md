---
name: material-counter
description: Use when the user asks who is up material or what the material balance is.
---

# material-counter
When to use: the user asks about the material count or who is ahead in material.
Steps:
1. Call list_pieces color=mine and color=theirs.
2. Sum the standard piece values for each side.
3. Report the net material difference.
Constraint: count only pieces on the board right now.
