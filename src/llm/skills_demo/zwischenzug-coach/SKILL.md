---
name: zwischenzug-coach
description: Use when the user asks about in-between moves before recapturing.
---

# zwischenzug-coach
When to use: the user asks whether an in-between move beats the obvious recapture.
Steps:
1. Call best_move to compare the recapture with alternatives.
2. Look for a more forcing reply that improves the order.
3. Show the zwischenzug and why it gains.
Constraint: the in-between move must be more forcing than the recapture.
