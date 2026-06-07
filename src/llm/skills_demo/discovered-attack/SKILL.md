---
name: discovered-attack
description: Use when the user asks about discovered attacks or discovered checks.
---

# discovered-attack
When to use: the user asks if moving a piece uncovers an attack.
Steps:
1. Call best_move top=3 to find moves that unveil a threat.
2. Identify the front piece and the unmasked attacker.
3. Give the move and what the discovery wins.
Constraint: ensure the moving piece itself stays safe or gains tempo.
