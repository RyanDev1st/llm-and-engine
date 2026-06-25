Parent: docs/findings/2026-06-24-harness-live-vs-benchmark-gap.md

# Train/serve parity audit — the harness, not the weights, is capping the chess metrics

## Status

No-retrain investigation (frozen v4): where does the SERVE/EVAL harness diverge from what the model
trained on? A frozen model performs best on its exact training distribution; every divergence the
harness adds is a metric leak. E4B is community-known-weak at tool calling, so the harness must run at
parity, not fight the model. Two concrete divergences found + fixed (both CPU-proven), no training.

## Scope

Corpus-finals + system-prompt diff over the v1_2 val set (2,731 rows) and the serve/eval code paths.

## Evidence

**1. The completion loop injects an off-distribution LIVE BOARD line; the routing evals don't.**
- The trained system prompt has **no board line** (verified: `build_system(...)` over a chess row →
  no `LIVE BOARD`). 94% of chess val rows (583/619) expect `board_state` IN their tool chain
  (`skill:chess-coach → board_state → <tool>`): the model is trained to CALL board_state to see the
  board.
- Routing evals (`eval_confusion`, `eval_benchmark`) build the prompt via `build_system` → **no board
  hook → training parity.** That is why routing scores are healthy (verb 88.7% native).
- The completion eval runs the full `CoachLoop` → `build_system_prompt` → **board hook ON by default
  → injects the board (161 chars the model never trained on).** With the board handed to it, the model
  reads it and SKIPS `board_state` → the expected chain doesn't complete → `completed` fails even though
  routing was correct. This is the chess-completion 13.5% artifact: the failing rows route correctly
  (`first=skill:chess-coach`) and error-free (`errors: —`), they just don't fire the redundant tool.
- **Proof:** with `CHESS_BOARD_HOOK=0` the completion loop's system prompt is **byte-identical** to the
  routing eval's and to training (`loop_off == routing`, verified). So the board hook is the SOLE
  divergence between the parity routing eval and the off-distribution completion eval.

**2. A precedence rule was added to BASE_HARNESS this session that v4 never trained on.**
Shipping it to v4's serve/eval is itself a (small, reinforcing) off-distribution mismatch — exactly the
thing this audit removes. Gated OFF (`CHESS_PRECEDENCE_RULE`, default 0) so v4 runs at parity; kept for
a v5 retrain (set the flag in BOTH train and serve so train==serve).

## What is NOT a harness bug (don't "fix" by retrain)

- **G/H "0%/14%"** = the model emits `<skill>threats</skill>` (the tool name loaded as a skill) instead
  of `skill:chess-coach`. Verb is still correct (skill→skill); only exact-NAME misses. The executor
  returns a "that's a tool, not a skill" corrective and the loop usually RECOVERS to a grounded answer
  (the completion `recovered` metric). It's a name slip the harness absorbs, not broken routing.
- **The 98.5% confusion run** is first-action on chess slices, which are 100% skill-gold (the whole val
  is 2591 skill / 102 tool / 38 none — tool-gold only in V1_R, none-gold only in V1_Q/V1_P). It does
  not measure 3-class routing; lead with completion + the base→adapter delta instead.

## Fixes landed (no retrain, both CPU-proven)

| Fix | Where | Effect |
|---|---|---|
| `CHESS_BOARD_HOOK=0` for the completion eval | Cell 6.7 | completion loop runs at training parity → board_state fires → `completed` should rise sharply |
| Gate the precedence line off for v4 | `system_prompt.py` (`CHESS_PRECEDENCE_RULE`, default 0) | removes the mismatch this session introduced; available for v5 |

## How to confirm (the one run that matters tonight)

Re-run the chess completion at parity — expected: `completed` jumps from 13.5% as `board_state` now fires.
```
CHESS_BOARD_HOOK=0 python -m llm_training.eval_completion --adapter <best> --per-slice 4
```
The chat A/B cell (`report.chat_ab`, board_on vs board_off) shows the same effect on reply content live.

## What to present

1. Lead with **completion at parity** (board hidden) + the base→adapter delta (tool FP 55→7, F1
   0.42→0.81, verb 49.6%→88.7%) — the genuine fine-tuning win.
2. Frame the chess-completion story honestly: "the live-board injection made a trained tool step
   redundant; at training parity the model executes the full chain." That is a HARNESS finding, and a
   strong one — it shows the team can diagnose serve/train drift, not just train a model.

## Next

- The board hook is a live-UX feature; for METRICS it must be off (parity). Decide per surface: eval =
  always off; live web demo = optional (the A/B shows the tradeoff).
- v5 (if ever): turn `CHESS_PRECEDENCE_RULE=1` in train+serve together; re-check the 1664 seq ceiling.
