---
name: king-safety
description: Use when the user asks how exposed a king is or whether to castle.
---

# king-safety
When to use: the user asks about king safety or castling.
Steps:
1. Call board_state to see castling rights and check status.
2. Call threats to test if the king is under real pressure.
3. Recommend castle, hold, or shelter the king.
Constraint: weigh safety against tempo; do not castle into an attack.
