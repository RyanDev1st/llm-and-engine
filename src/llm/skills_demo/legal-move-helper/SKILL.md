---
name: legal-move-helper
description: Use when the user asks which moves are legal for a piece or from a square.
---

# legal-move-helper
When to use: the user asks what a piece can legally do.
Steps:
1. Call legal_moves with the square in question.
2. List the legal destinations clearly.
3. Flag any that walk into a tactic.
Constraint: report only moves legal_moves returns.
