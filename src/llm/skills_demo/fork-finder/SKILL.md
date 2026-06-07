---
name: fork-finder
description: Use when the user wants to find a fork — one piece attacking two targets at once.
---

# fork-finder
When to use: the user asks if there is a fork available.
Steps:
1. Call threats and best_move top=3 to spot double attacks.
2. Check whether a knight or queen hits two targets.
3. Give the forking move and what it wins.
Constraint: confirm both targets are real before calling it a fork.
