Parent: [2026-06-25-train-serve-parity-audit.md](2026-06-25-train-serve-parity-audit.md)

# Kaggle chat A/B (board_on vs board_off) + the board_state-empty fix

**Status:** Measured (the 2026-06-25 Kaggle chat run, E4B v4 adapter, nf4). Two serve fixes landed +
CPU-tested; both await live re-verification on the next run. **Scope:** the chat A/B compared the serve
prompt with the LIVE BOARD hook ON vs OFF on identical hand-written prompts; three problems surfaced.

## What the run showed (verbatim observations)

1. **board_on HALLUCINATES — board_off is better.** With the LIVE BOARD line injected, the model gives
   off-distribution, less-apt replies (it was trained with NO board line). board_off produces the apter
   answer. This **empirically confirms** the parity audit's prediction (board hook off = trained shape).
2. **board_off sometimes plays an EXTRA move.** "play e4 for me" → the model played `e4` **and then an
   unrequested `Nc6`** (playing both sides). board_on stopped after `e4`.
3. **board_off is slower** (e.g. that turn: 80.8s vs 43.2s) — because of the extra loop steps in #2.
4. **`board_state` returned EMPTY.** The transcript shows `<tool>board_state fields=<['all']></tool> ->
   board_state:` (nothing after the colon).

## Root cause — #4 is the trigger for #2 and #3

`ToolExecutor._board_state` (backend/tools.py) matched the `fields` value literally. The model emitted
`fields=<['all']>` (schema-placeholder `<…>` + python-list `['all']` junk). `"<['all']>"` equals
neither `"basic"` nor `"all"`, so **no field matched and it returned a bare `"board_state:"`**.

That empty result is also why board_off looped: with the board hidden (hook off, as trained), the model
**correctly** calls `board_state` to see the result of its move — but got nothing back, stayed
ungrounded, and flailed into playing `Nc6`. board_on didn't loop only because it could read the injected
board. So a working `board_state` is what lets board_off both ground AND stop. **#4 fixed ⇒ #2/#3 fixed.**

## Fixes landed (CPU-tested)

- **board_state never returns empty** (`tools.py _board_state`): liberal parse — strip wrapping
  `<>[]'"` from each field token so `<['all']>`/`[all]` still resolve to `all`; and an unrecognized
  value falls back to `basic` instead of empty. Test: `test_tool_validation.py::
  test_board_state_never_returns_empty_on_junk_fields`.
- **Board hook default flipped OFF** (`inference.py _BOARD_HOOK` default `"1"`→`"0"`; serve notebook
  `colab_serve_e4b.ipynb` BOARD_HOOK `"1"`→`"0"`). This makes the LIVE serve match training distribution
  — the empirically-better config. `CHESS_BOARD_HOOK=1` still restores the injected board for an A/B.
  Test updated: `test_plugins.py::test_build_system_prompt_injects_board_not_skill_body` enables the hook
  explicitly (it tests the injection mechanism, not the default).

## Not changed (deliberate)
- `ask_chessbot` returning a canned KB line ("That's a great chess question…") is the knowledge-base
  fallback, not a bug; the model *chose* it for a "positional assessment" (eval would've been apter) —
  a routing choice, not a serve fault.
- The `list_pieces color=…` / `undo no moves` errors in the completion log are correct corrective
  errors (validation working); the loop recovers.

## Next (verify on the next Kaggle run)
- Re-run the chat A/B with board OFF as default: confirm `board_state` returns a real position, the
  "play e4" turn no longer plays `Nc6`, and the turn is faster.
- The completion-at-parity (chess) number is now unblocked twice over (board hook off **and**
  board_state grounding restored).
