---
name: position-evaluator
description: Use when the user asks who is better, by how much, or to assess the position.
---

# position-evaluator
When to use: the user asks for an evaluation of the current position.
Steps:
1. Call eval to get the engine score from White's POV.
2. Translate the score into plain advantage terms.
3. Name the one factor driving the evaluation.
Constraint: report the sign and size as the engine gives it; do not inflate.
