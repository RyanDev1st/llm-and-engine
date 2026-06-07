---
name: perpetual-check
description: Use when the user asks about forcing a draw by perpetual check or repetition.
---

# perpetual-check
When to use: the user is worse and asks whether a perpetual draw exists.
Steps:
1. Call best_move to see if a repeating checking line scores 0.
2. Confirm the checks actually repeat the position.
3. Give the first check of the perpetual.
Constraint: claim a perpetual only if the checks truly repeat.
