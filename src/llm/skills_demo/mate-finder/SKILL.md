---
name: mate-finder
description: Use when the user asks if there is a forced checkmate or a mating attack.
---

# mate-finder
When to use: the user asks whether a forced mate exists.
Steps:
1. Call best_move to see if the engine reports a mate score.
2. If mate exists, give the first move and the mate distance.
3. If not, say so and give the best practical try.
Constraint: only claim mate when the engine returns an M-score.
