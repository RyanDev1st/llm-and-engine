---
name: pin-detector
description: Use when the user asks about pins — a piece stuck in front of a more valuable one.
---

# pin-detector
When to use: the user asks whether a pin exists or how to exploit one.
Steps:
1. Call board_state with fields=fen to see the alignment.
2. Identify the pinned piece and the piece behind it.
3. Suggest how to pile on the pinned piece.
Constraint: verify the line of the pin on the actual board.
