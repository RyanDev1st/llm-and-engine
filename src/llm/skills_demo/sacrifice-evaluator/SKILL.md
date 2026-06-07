---
name: sacrifice-evaluator
description: Use when the user wants to know if a sacrifice is sound.
---

# sacrifice-evaluator
When to use: the user asks whether giving up material works here.
Steps:
1. Call eval on the position after the sacrifice.
2. Use best_move series to read the follow-up line.
3. Say sound or unsound with the reason.
Constraint: call a sac sound only if the engine line supports it.
