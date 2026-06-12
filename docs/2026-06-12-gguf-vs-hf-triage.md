Parent: 2026-06-12-model-audit.md

# GGUF vs HF — speed/fidelity triage (GPU benchmark)

## Status

Ran the same 6-turn multi-turn audit on **GGUF (Q4_0, full GPU offload, prefix cache
on)** and compared to the earlier **HF (bnb nf4 4-bit + LoRA)** runs. Headline: **GGUF
is ~2.4× faster but Q4_0 fabricates eval numbers in narration** (the coarser quant
degrades number fidelity). Routing, persistence, coverage, streaming all work on both.
Services left running for triage (see end).

## Evidence — latency

Same prompts, single product path (`variant=sft`, coverage on):

| Turn | HF (nf4) | GGUF (Q4_0) |
|---|---|---|
| T1 play e4 (move) | 5.2 / 9.6s | 9.4s |
| T2 eval | 14.2 / 7.0s | **4.0s** |
| T3 3 best moves | 20.5 / 21.1s | **6.3s** |
| T4 threats | 19.3 / 20.4s | **5.2s** |
| T5 recall (no tool) | 7.5 / 6.4s | 3.0s |
| T6 undo | 10.6 / 9.7s | 4.0s |
| **mean** | **12.4–12.9s** | **5.3s** |

GGUF ≈ **2.4× faster**. The multi-tool turns (T3/T4) gain most — GGUF 5-6s vs HF ~20s —
which is exactly where the prefix cache + faster decode compound. T1 (9.4s) is the
cold-cache first turn.

## Evidence — fidelity (the real finding)

GGUF Q4_0 **fabricates the eval number** in its own prose. Direct capture:

```
RAW tool result : score: +0.37 pawns from white POV, depth=18
GGUF reply      : "The current position is slightly better for white (-0.18).
                   The position stands at +0.37 pawns from white POV ..."
```

The model invented `-0.18` (wrong value AND wrong sign), then the answer-coverage layer
appended the grounded `+0.37` — so the user sees a **contradiction** (two different
numbers). More cases in the audit: T2 "(-0.39)" vs appended "+0.42"; T3 listed all three
black replies at the same `-0.39`; T4 invented "Nc3+ ... parried by d6" (the threats tool
returns only the best opponent move + score, no parry).

HF (nf4) did **not** show this — its T2 read cleanly: "White is up a slight edge by about
0.42 pawns." So this is a **quantization-fidelity regression specific to Q4_0**, not a
harness bug. Q4_0 is a coarse legacy quant; the merged-adapter weights lose the precision
to reproduce the small signed eval numbers the model was trained to echo.

Note: answer-coverage **did** fire and ground the reply (the real number is present), and
nothing wrong reached the board (tools ran on real state). The damage is cosmetic-but-
serious: a visible contradictory number undermines the "grounded, never fabricates" pitch.

## What works on both backends (unchanged)

- Multi-tool routing 5/6 turns; board persistence across turns; undo reverts; coverage
  forces the required tools; streaming (SSE) delivers steps live; plugin prompt-start hook
  injects the live board (no board_state call); progressive disclosure intact.
- Memory recall (T5) weak on both — model-bound (E2B misreads "this game"), not quant.

## Recommendation (for triage)

The speed win is large and real. The fix for the fidelity regression is small and
deterministic — don't switch back to slow HF over a cosmetic number bug:

1. **Number-consistency guard (recommended, deferred item now justified).** Deterministic,
   no model call: regex the eval/score number the model wrote; if it doesn't match the
   tool result's number, replace it with the real one (or drop the model's number and keep
   only the grounded sentence). Kills the contradiction at the source. ~1 small function +
   tests. Pairs with the existing answer-coverage.
2. **Higher-fidelity GGUF quant.** Re-export the merged adapter at **Q5_K_M or Q6_K**
   instead of Q4_0 — markedly better number fidelity, slightly larger/slower (still far
   faster than HF). Best long-term if VRAM allows on the 4060.
3. **Keep HF for fidelity** — only if 1+2 somehow fail; costs the 2.4× speed.

Best combo: **GGUF (Q5_K_M) + number-consistency guard** — fast AND grounded.

## Update — fixes shipped + Q5_K_M re-audit

Both planned fixes landed and were live-verified:

- **Number guard** (`fd952c40`): deterministic — replaces a fabricated eval number with
  the real tool value (conservative: single unmatched number vs single eval source;
  best_move scores / move SANs never touched). 8 tests.
- **Q5_K_M re-export** (`bab3127b`, default `20675108`): re-quantized the merged adapter
  at Q5_K_M (~3.6 GB). Now the serving default; `CHESS_GGUF_PATH` overrides.

**Q5_K_M 6-turn re-audit vs the earlier runs:**

| | HF nf4 | Q4_0 | **Q5_K_M** |
|---|---|---|---|
| mean latency | 12.4s | 5.3s | **5.5s** |
| eval interpretation (T2, eval +0.42) | clean | **"Black is slightly better" ✗ (inverted)** | **"White is slightly better" ✓** |
| memory recall (T5 "first move?") | failed | failed | **"e4" ✓** |
| routing / persistence / undo | ✓ | ✓ | ✓ |

Q5_K_M fixed the eval-interpretation inversion **at no speed cost** (still ~2.3× faster
than HF). Recall improved too (the live board hook surfaces last_move).

**Residual (E2B narration ceiling, not quant):** on a compound best-moves ask the model
still fabricates the move-**name** list in prose (said d5/Nc6/g6; answer-coverage appended
the real e5), and embellishes threat rationale. The number guard covers eval numbers, not
move names; answer-coverage still grounds the key fact beside the drift. The real fix for
this is reasoning-SFT / E4B (deferred), not quantization.

**Net:** ship **Q5_K_M + number guard + answer-coverage** — fast, eval-grounded, with the
truth always present. Recommendation from the top of this doc is now implemented.

## Services running (for your triage)

- `:7861` GGUF model service (GPU, pid was 34504) — holds the weights.
- `:7862` weightless app on current code → the GGUF service. Open `http://127.0.0.1:7862`.
- `:7860` your earlier weightless app (also points at :7861, so now GGUF too).
Kill the GGUF service to free the GPU: `taskkill /F /PID <pid on :7861>`.
