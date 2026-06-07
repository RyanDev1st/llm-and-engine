---
name: knight-outpost
description: Use when the user asks where to plant a knight or about outposts.
---

# knight-outpost
When to use: the user asks about a strong knight square or outpost.
Steps:
1. Call board_state with fields=fen to find a protected hole.
2. Pick the square no enemy pawn can challenge.
3. Give the route that lands the knight there.
Constraint: an outpost must be pawn-supported and unchallengeable.
