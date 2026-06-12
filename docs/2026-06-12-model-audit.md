Parent: none

# Live model audit — multi-turn tool use, persistence, memory

## Status

Audited the trained adapter live against the running server (`:7860`, HF adapter via
the model service) with a scripted 6-turn conversation (`A:/Download/audit_live.py`).
Verdict: **tool routing + grounding are strong; board persistence works; conversation
memory infra works but E2B recall is weak; one tagless-leak bug found and fixed.**

## Scope

Single-engine product path (`variant=sft`, `coverage=on`). Six turns: move → eval →
best-moves → threats → memory recall → undo. Plus a focused memory probe.

## Evidence

Conversation (reset → 6 turns), `tools` = tools actually executed that turn:

| Turn | User | Tools | Reply (grounded?) | Latency |
|---|---|---|---|---|
| 1 | play e4 | `move` | "That was e4…" — board → `history=[e4]`, turn=black ✓ | 5.2s |
| 2 | what's the evaluation now? | `eval` | "+0.44 from white" — sees the e4 board ✓ | 14.2s |
| 3 | give me the 3 best moves for black | `best_move` | "e5, c6, c5" — correct ✓ | 20.5s |
| 4 | any threats? | `threats` | "d4, +0.97 for them" ✓ | 19.3s |
| 5 | what was my very first move? | — | "I do not have access to a history…" ✗ recall | 7.5s |
| 6 | undo my last move | `undo` | "e4 has been undone" — board → `history=[]` ✓ | 10.6s |

Latency: **mean 12.9s, min 5.2s, max 20.5s**.

**Multi-turn tool use — PASS.** 5/6 turns routed to the correct tool, every analysis
grounded in the real engine number (e4 = +0.44; black replies e5/c6/c5; opponent threat
d4 = +0.97). No fabrication, no tag leaks in the main flow.

**Board persistence — PASS.** The board carried across all turns: e4 stayed live through
eval/best_move/threats, and undo correctly reverted it to the start. Server-side game
state, independent of the model's text memory.

**Conversation memory — infra PASS, recall WEAK.** A focused probe (reset → play e4 →
"what move did I just play?") returned `turns_total=2, turns_kept=2, evicted=0` — the
model **does** receive prior turns; the "0 kept" seen on turn-1 in the UI is correct
(empty history). The T5 recall miss is E2B being weak: it misread "this game" as "previous
games" and declined. This is model-bound (a reasoning/recall limit), not an eviction bug,
and is the kind of thing the future E4B/reasoning-trained step improves.

**Bug found + fixed — tagless bare-call leak (`c81128b5`).** The memory probe's reply was
literally `review_move depth=1` — a tool call the model emitted with NO `<tool>` tags, so
`extract_call` didn't recover it and `contains_tool_call` didn't flag it → it leaked into
chat. Fixed: when the whole reply is a known tool name + ≥1 `k=v` arg, recover it to a
canonical call so it executes (args required, so prose can't false-match).

## Speed

Each turn is ≥2 model generations (decide a tool, then narrate) at ~6-7s each on the 4060
(E2B 4-bit). Single-intent turns ≈ 13s; multi-tool turns (best_move/threats) ≈ 20s. The
worst case is the model's occasional `board_state` + `load_skill` *prefix* before the real
analysis (seen on the compound opening prompt in the UI) — that adds ~2 generations
(~14s). Note: in this audit NO turn did that prefix — focused turns were efficient (one
tool each).

Levers, honest (serve-side, no retrain):
1. **Eliminate the board_state/load_skill prefix** when it happens — inject cheap board
   facts into the system prompt (model stops calling `board_state`), and/or pre-load the
   always-on `chess-coach` skill so it skips `load_skill`. Biggest variable win (~14s on
   those turns). Tradeoff: pre-loading the coach skill softens strict progressive
   disclosure for that one always-relevant skill.
2. **Trim the final narration budget** (160 → ~120 new tokens).
3. **GGUF q4_0** may decode faster than HF 4-bit on the 4060 (HF is the proven path; would
   need a head-to-head).
4. The ~6-7s/generation floor is E2B-on-4060; only a smaller/faster quant or model moves it.

The dominant cost is `N generations × ~6-7s`. Coverage trades a generation for
completeness; that's the design choice. The model itself is accurate and well-grounded —
the limits are recall depth (model-bound) and raw decode speed (hardware/quant-bound).

## Next

1. Decide on speed lever #1 (board-facts injection + optional coach pre-load) — the only
   large serve-side win. (Needs a yes on the progressive-disclosure tradeoff.)
2. Recall depth and novel-phrasing routing are the E4B/reasoning-trained items.
3. Bare-call leak fix shipped; re-audit after the next app restart to confirm in the flow.
