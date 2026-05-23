Parent: docs/superpowers/specs/2026-05-23-chess-coach-sft-design.md

# Dataset rebuild findings (v3)

## Status
Complete.

## Scope
Rebuilt chess_assistant_v3 train/val from human slices.

## Evidence
- Human records cleaned: 3185 {'A': 630, 'E': 350, 'F': 315, 'G': 140, 'B': 385, 'C': 280, 'H': 210, 'I': 420, 'J': 280, 'K': 175}
- Dropped duplicate `slices/slices/slice C.json` (colder tone); kept ex_C variant.
- Stripped ' (x_N)' artifact from 1,750 user turns (slices B,C,H,I,J,K).
- Warmed slice-C error narration; canonicalised system prompt (dash mojibake fixed).
- Authored slice D (implicit eval, real Stockfish depth-15 scores): 315.
- Exact-dedup removed: 1229.
- Validation rejects: 0.
- Final per-slice: {'A': 630, 'B': 296, 'C': 244, 'D': 308, 'E': 350, 'F': 315, 'G': 140, 'H': 139, 'I': 119, 'J': 122, 'K': 238}
- Split: train=2616 val=285 total=2901.

## Next
1. Smoke train, then full 3-epoch QLoRA on gemma4_e2b.
2. Routing-accuracy audit on val.
