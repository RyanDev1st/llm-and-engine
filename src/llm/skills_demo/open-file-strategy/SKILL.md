---
name: open-file-strategy
description: Use when the user asks how to use open or half-open files with rooks.
---

# open-file-strategy
When to use: the user asks about rooks, open files, or file control.
Steps:
1. Call board_state with fields=fen to find open files.
2. Plan to double rooks on the most useful file.
3. Name the entry square the rooks aim for.
Constraint: contest a file only when you can win or hold it.
