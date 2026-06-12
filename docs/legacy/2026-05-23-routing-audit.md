Parent: docs/superpowers/specs/2026-05-23-chess-coach-sft-design.md

# Routing-accuracy audit

## Status
Overall tool-routing accuracy: 276/285 = 96.8%

## Scope
Adapter: `../../runs/gemma4_chess`. Validation set: 285 conversations.

## Evidence
Per-slice routing accuracy:
- A: 61/63 = 97%
- B: 29/29 = 100%
- C: 23/24 = 96%
- D: 30/30 = 100%
- E: 35/35 = 100%
- F: 29/31 = 94%
- G: 13/14 = 93%
- H: 13/13 = 100%
- I: 11/11 = 100%
- J: 9/12 = 75%
- K: 23/23 = 100%

Mode-2 discipline: 266/266 clean (0 records emitted a <tool> after a tool result).

## Top routing confusions
- 3x J: gold=None pred=ask_chessbot
- 2x A: gold=move pred=None
- 2x F: gold=review_move pred=None
- 1x G: gold=threats pred=None
- 1x C: gold=move pred=review_move

## Next
1. Export merged adapter to Q4_0 GGUF.
2. Wire adapter into the web app and run end-to-end.
