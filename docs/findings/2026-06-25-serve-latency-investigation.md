Parent: [reference/harness-system-overview.md](../reference/harness-system-overview.md)

# Serve latency — what's actually slow, and the real levers

**Status:** Investigated (static + code reading). One safe loop optimization shipped (verify-probe
gating); the rest are ranked recommendations with a measured A/B. **Scope:** live turns take 20–90s
(real transcripts); the user asked to make the runtime faster and optimize the thinking loop.

## The headline reframe: latency is DECODE-bound, not prefill-bound

Turn latency ≈ **(decode steps per turn) × (tokens per step) × (T4 seconds/token)**. The prompt
prefill is NOT the bottleneck:

- **KV-cache reuse is the wrong lever here.** `kv_cache.py`'s own docstring: reuse "only ever saved
  prefill (~tenths of a second), never decode (the real cost)." It's `CHESS_KV_REUSE=0` by default,
  and it only ever ran on the *non-streaming greedy* path (`_gen_cached`). The **live UI path is
  streaming** (`_gen_stream`), which never touches the cache — so the cache is dead code live, and
  even enabling it would save tenths of a second against 20–90s turns. Not worth pursuing.
- **Stockfish depth is not the lever either.** All **1353/1353** analysis calls in the corpus emit
  `depth=` explicitly, so the model never relies on `DEFAULT_EVAL_DEPTH` — lowering it does nothing,
  and capping it would override the depth the model asked for to shave ~1s of Stockfish against
  ~10s+ of decode per step. Decode dominates.
- The two correctly-applied fixes already in place — eos includes the turn-ender + pad (no `<pad>`
  flood) and a version-robust early-stop `StoppingCriteria` (no run-to-cap) — are why a *single*
  short reply is ~10–20s and not worse. They're load-bearing; keep them.

So the only ways to go faster are **fewer decode steps**, **fewer tokens per step**, or a
**faster-decoding model** — caching and Stockfish tuning don't move the needle.

## Where the decode steps go (the thinking loop)

A simple analytical turn ("how am I doing?") is **2 decodes minimum**: step 1 decides
(`<goal><think><tool>eval…`), step 2 narrates the result. That floor is inherent — the model needs the
tool result before it can narrate. The 20–90s *variance* comes from EXTRA decodes the loop adds:

| Source | Extra decodes | Lever |
|---|---|---|
| `_verify_fulfilled` probe on skill-load turns | +1 every coach/puzzle turn | **gated this turn** (see below) |
| coverage force-routing + "Wait" steers | +1 per forced tool | `CHESS_THIN_HARNESS=1` |
| `_force_answer` / `_force_synthesis` | +1 on deflections | `CHESS_THIN_HARNESS=1` |
| error-recovery loops (bad FEN retried, tagless calls) | +N until it gives up | the `new_game` + clearer-error fixes (PR #20) |
| auto-mode `<think>` tokens per step | more tokens, same steps | inherent (the trace IS the product — do NOT cut) |

The 91.9s "reset board" turn was the worst case: repeated `load_fen`-with-a-bad-FEN decodes. PR #20
collapses that to one `new_game` call + a narration — the bug fixes are also latency fixes.

## Shipped this turn — gate the verify probe (one fewer decode per coach turn)

`_verify_fulfilled` was a FULL extra generation firing on EVERY skill-load-only turn, even when the
draft was already a good answer. Now it fires only when the draft LOOKS like a non-answer (a
deflection blurb, an ask-back, or < 40 chars). A confident substantive answer is trusted as-is. The
bad cases still route through the same probe + deflection handling, so their behavior is unchanged —
verified by `test_verify_gating.py` (good answer skips the probe; deflection still caught) plus the
existing serve/coverage suites (56 green). Disabled entirely under `CHESS_THIN_HARNESS=1`.

## The ranked levers (recommendation)

1. **`CHESS_THIN_HARNESS=1` — biggest SAFE win.** Removes the coverage force-route decodes, the verify
   probe, and the force-answer/synthesis decodes. On coach/puzzle/multi-tool turns that's often 1–2
   fewer full generations. Already built + tested; A/B it live, then flip the default if it holds.
2. **GGUF Q4_0 — biggest decode-SPEED win (~2.4× tok/s, memory `gguf-q4-fabricates-eval`).** The
   serve runs HF nf4 for fidelity; Q4_0 decodes ~2.4× faster. Its known eval-number fabrication is
   already mitigated by the `_correct_eval_number` guard. This is the one architectural change that
   speeds EVERY decode, not just trims steps — worth a measured trial.
3. **The PR #20 bug fixes** already cut the error-recovery decode loops (the 90s outliers).
4. **Do NOT** default to fast mode (deletes `<goal>`/`<think>` — the reasoning trace is the product;
   memory `fix-intentionally-not-by-sacrificing-features`), and do NOT chase KV reuse or Stockfish
   depth (proven non-levers above).

## How to measure (already wired)

`model_hf.py` logs `[gen] in=… out=… Ns (X tok/s)` per call (`CHESS_GEN_TRACE=1`, default on). Read
`model_server.log` to see, per turn: how many `[gen]` lines (decode steps), tokens each, and tok/s.
That tells you decode-vs-Stockfish split and whether thin mode / GGUF actually moved it — measure,
don't guess.
