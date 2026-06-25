Parent: docs/findings/2026-06-24-harness-live-vs-benchmark-gap.md

# Answer-quality triage — serve fixes + the corpus finding that decides a v5 retrain

## Status

The first full Kaggle report run (2026-06-24, E4B v4 nf4) completed end to end (clean-exit fix held).
The metrics looked strong (OOD STRESS completion 91.7%, grounded 95%, routing verb 88.7%) but the
thing that matters — **the final answer the user reads** — was frequently unhelpful or irrelevant.
This doc separates what was a SERVE bug (fixed here, CPU-verified) from what is a DATA problem (a
concrete, evidence-backed retrain target). It does NOT itself trigger a retrain; it scopes one.

## Scope

Re-read of the 14 captured showcase turns (bare harness + web sandbox) + a corpus-finals analysis of
all 72,329 v1_2 train rows. No GPU.

## Evidence — the live answer failures, classified

| # | mode | example (live) | cause |
|---|---|---|---|
| 0 | broken catalog | "what opening?" / "gimme a puzzle" → `unknown_skill` | **my showcase bug** — ran `plugin_context=None`; fixed `93232354` (web-app parity) |
| 1 | broken reply | "play e4 for me" → `Coach: <` | SERVE — `move e4` (no `san=`) → corrective error → `<` fragment reached the user |
| 2 | irrelevant tool | "keep hanging my queen" → ran `threats` on an empty board | model choice (NOT coverage-forced); DATA/skill-body |
| 3 | raw-result leak | "...for the set time. `breathing_timer: 120s set — about 6…`. How are you feeling?" | SERVE — model parroted the executor's raw result line |
| 4 | wrong substance | "is my queen safe on a5" → confused, unverified | DATA/grounding |
| 5 | narrate-and-deflect | "i'm dogwater at endgames" → "What part of the game are you focused on?" | DATA (see corpus finding) |

## Part A — serve fixes (landed, CPU-verified, `test_answer_quality_guards.py` 10 tests)

1. **move arg coercion** (`toolfmt.parse_call`): `<tool>move e4</tool>` / `move Nf3` / `move O-O` /
   `move e2e4` now fill `san=` when it's a CLEAN single move token (`fullmatch`). `move rook f8`
   (multi-word, can't-spawn-a-piece) still returns the corrective error — guarded by the existing
   `test_reset_and_errors` expectation, which caught an over-broad first cut.
2. **degenerate-final guard** (`inference._is_markup_fragment`): a markup-only / 1-char reply (a lone
   `<`, a dangling tag) is now treated as a whiff → the answer-retry / fallback fires instead of
   shipping `Coach: <`.
3. **result-echo strip** (`inference._strip_result_echo`, in `_finalize`): an EXACT verbatim echo of a
   single-line `name: payload` tool result is removed from the final (exact-match only, never touches
   errors, multi-line skill bodies, or normal prose).

These fix #1 and #3 outright. They do NOT fix #2/#4/#5 — those are not serve bugs.

## Part B — the corpus finding (this is what decides a retrain)

Final-reply analysis of all 72,329 train rows (serve detectors `_is_ask_back`/`_is_deflection` reused):

- **Closer monoculture.** **70.2%** of all finals end with a question; **100%** in the chess analysis
  slices (A,B,D,E,F,G,H) AND the largest slice V1_O (18,121 rows, cross-domain routing). The trained
  closer is "grounded statement + binary offer" ("Qxc5 is the move… Should I go deeper, or look at the
  alternatives?"). Good in isolation, but so ubiquitous that EVERY answer bounces a question back —
  which reads as deflection even when grounded. (Serve `_is_ask_back`=0.0%, `_is_deflection`=0.6%: the
  data does NOT teach the bad blurb-deflection; it teaches the relentless offer-closer.)
- **Process-narration finals.** **V1_N_human_chat_skill_bridge: 71.5%** (1,868 / 2,611) of finals
  NARRATE harness mechanics instead of answering — e.g. user "am i cooked or is there a move here?" →
  "I used the helper output to identify chess intent, then loaded chess-coach for board-safe help.
  Want the detail?" This is failure #5 verbatim, and it lives in the data on exactly the casual-chat
  prompts that failed live. V1_A/V1_B ~25% show a milder skill-selection-narration variant.
  (V1_C at 100% is a FALSE POSITIVE — its task is to CONFIRM dynamic-tool use, so "I used tool_zb_951
  from the current manifest" is the correct grounded answer; excluded.)

## Recommendation — a targeted v5 IS justified, but scoped to the data, not the harness

A retrain is warranted because #5 (and the closer monoculture behind the "feels like deflection"
complaint) has a clear, quantified DATA root that no serve regex can fix safely. The targeted change:

1. **Rewrite V1_N finals answer-first** — no "I ran the helper / loaded chess-coach" process
   narration; answer the user's stated question, then (optionally) offer. ~1,900 rows.
2. **Break the closer monoculture** — mix in finals that answer and STOP (no trailing offer) across
   the chess analysis slices + V1_O; target the offer-closer at ~40–50%, not ~100%, so a direct
   question gets a direct answer.
3. Keep the part-A serve fixes (orthogonal; they make the next run cleaner regardless).

Explicitly NOT in scope: #2 (irrelevant board tool on a no-board question) and #4 (wrong substance)
are routing/grounding, better attacked first via the chess-coach skill body ("don't run a board tool
for a general/no-board question"; "verify before claiming") — the frozen-model lever — before adding
corpus rows for them.

## Next

- Confirm the next clean Kaggle run (plugin-context fixed + part-A serve fixes) before authoring v5
  data — the showcase will look materially better and may re-scope #2/#5.
- If the narrate-and-deflect persists on V1_N-shaped casual prompts post-fix, author the v5 data
  change above and retrain (deliberate, against this confirmed signal).
