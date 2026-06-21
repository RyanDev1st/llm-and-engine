Parent: docs/reference/harness-system-overview.md

# Peer-review gap verification + remediation plan (2026-06-22)

**Status:** verification complete — all 6 reviewer gaps reproduced against current source (file:line
evidence below). Remediation plan proposed; **not yet executed** (awaiting go-ahead on the
behaviour-risk items #1 and #4).
**Scope:** an external peer review of `docs/reference/harness-system-overview.md` returned 6 gaps. We
verified each in the live code (codegraph + targeted reads), confirmed the reviewer's "not a gap"
list is also accurate, and turned the result into a prioritized fix plan.

## Verification verdicts (all TRUE)

1. **Chess coverage over-forces OOD prompts — the biggest real harness gap.**
   `src/llm/backend/inference.py:651` appends `routing_hints(user_message, game_over)` and `:661`
   builds `required = matched_calls(user_message)` — both **unconditionally**, guarded only by
   `game_over`. `src/llm/backend/tool_hints.py:47-59` the `eval` trigger matches `how am i doing`,
   `how'?s my position`, etc. So "how am I doing with my taxes?" → `matched_calls` returns
   `{eval: <tool>eval depth=18</tool>}`, and the loop's coverage backstop (`inference.py:768-771`)
   **force-routes and EXECUTES it** even when the turn is non-chess. `routing_hints` (a prompt nudge)
   is lower-risk; the coverage **force-execute** is the dangerous half. No plugin-context / manifest
   gate exists. **Confirmed runtime bug.**

2. **Evaluation is first-action routing only.** `eval_benchmark.py::_bench` and
   `eval_confusion.py` score the first action's verb+name. They do NOT score: arg correctness,
   executor success (did the tool return non-error?), final-answer grounding (does the reply cite the
   result?), multi-turn task completion, or harness recovery. `bench_misses.py` now logs *what* was
   emitted, which helps, but the headline metric is still routing, not completion. **Measurement gap,
   not a runtime bug** (already partially mitigated by miss-logging + the built native-mode probe).

3. **Stress/OOD eval is too small.** `bench_suites.py::_CASES` = 20 hand-written rows. Useful as a
   robustness smoke test; too small to claim broad unseen-domain robustness (wide CI at n=20). **TRUE.**

4. **GGUF decode drifts from the trained/HF path.** `model_gguf.py:63` (`generate`) and `:85`
   (`generate_stream`) both hardcode `top_p=0.9, repeat_penalty=1.2`. The HF serve path runs decode
   penalties OFF (commit `09d209eb` — "penalties corrupt skill/tool name copying"). `repeat_penalty`
   reweights logits even under greedy/temp-0 decode, so it can flip the argmax on a repeated NAME
   token — the same failure that motivated turning penalties off on HF. Consistent with memory
   `gguf-q4-fabricates-eval`. **Real train/serve fidelity risk.**

5. **Frontend board reconcile has chess-state edge cases.** `index.html:981` `sameBoard` compares
   only FEN fields [0] (placement) and [1] (turn) — **ignores castling rights [2] and en-passant
   [3]**, so a backend FEN differing only there reads as "unchanged" → no reconcile → desync.
   `:757-779` `makeMove` line 763 **always promotes to queen** (`{type:'q'}`); `:989-1000`
   `tryApplyUci` parses `uci[0..3]` and **drops `uci[4]`** (the promotion piece, e.g. `e7e8n`). So an
   underpromotion is applied as a queen and lost from history. **TRUE** — display/demo correctness,
   does not touch the trained model or the benchmark.

6. **Generic tool failures mislabeled as engine failures.** `tools.py:113-118` `execute()` wraps
   `_dispatch` in `try: … except chess.engine.EngineError: return "error: engine_unavailable"` **and**
   a bare `except Exception: return "error: engine_unavailable"`. Since `_dispatch` also routes plugin
   tools and the `python`/sandbox tool, ANY non-chess tool bug is reported to the model (and user) as a
   Stockfish outage. **TRUE.**

**Reviewer's "not a gap" list — also accurate (re-verified):** `<think>`/`<goal>` reply leak is
fixed (`_split_reasoning` in `_finalize`, `inference.py:607`); direct-answer-as-plan-panel is fixed
(`is_plan_panel` requires a `<plan>`); reasoning mode is threaded (`respond → build_system_prompt`);
the symmetric tool-as-skill corrective error is implemented (`tools.py:170-181` + `_load_skill`);
memory/context-window/KV reuse have reasonable harness-side guards. The reviewer is credible.

## Second review (2026-06-22, richer) — additional verified gaps + the unifying reframe

A second peer review re-confirmed the 6 above and added more, all **verified true** in source. The
key insight: most OOD harness gaps share **one root cause — the deterministic layer is chess-
hardcoded and should read the live manifest (`serving_tool_manifest(plugin_context)`), not a baked-in
official list.** New verified items:

- **A. Recovery is not plugin-aware.** `inference.py:351` `_TOOL_NAMES = official_tools() |
  compute_tools() | {load_skill}` (module-level constant), and `_MALFORMED`/`_ECHO`/`_BARE`
  (lines 358-367) derive their name alternation from it. So malformed/tagless recovery in
  `extract_call` (line 370) works for chess tools but NOT plugin tools — a perfectly-wrapped
  `<tool>convert_units …</tool>` runs (the `"<tool>" in s` path is name-agnostic), but a tagless
  `convert_units value=5 …` is not recovered. **TRUE.**
- **B. Plugin results are not grounding-enforced in final answers.** `_result_signal` (line 215) only
  recognizes chess prefixes (`score:`/`best:`/`review:`) → returns None for `breathing_timer: …` /
  `convert: …`; and `_ensure_required_narrated` (line 233) iterates only the chess `required` set
  (from `matched_calls`). So a plugin tool's grounded result is never enforced. The captured breathing
  turn proves it: tool returned `breathing_timer: 10s set …`, final answer dropped it. **TRUE.**
- **C. Tool-as-skill corrective error lacks the arg schema.** `tools.py _load_skill` returns
  `… call it with <tool>{name} ...</tool>` — a literal `...` placeholder, no live arg schema, so the
  model guessed `seconds=10`. Should emit `<tool>breathing_timer seconds=<seconds></tool>` from the
  manifest. **TRUE.**
- **D. `applies_when` is prompt text, not a runtime gate.** `serving_tool_manifest:509` returns all
  official tools unconditionally; `applies_when` (game_in_progress / has_history) is rendered into the
  prompt but the manifest is not dynamically hidden/disabled per board state (runtime only validates
  required-args/enums via `validate_call`). A harness-contract gap if strict schemas/hook gates are
  wanted. **TRUE (lower priority — a design choice, not a bug).**
- **E. Plugin-local arg extraction (optional).** Obvious deterministic OOD cases ("scale … 12 up to
  30" → `scale_recipe from_servings=12 to_servings=30`) could be filled by **plugin-local** matchers —
  explicitly NOT global regex. A nice-to-have that reduces a model weakness without retraining.

The second review's other items map to the first six: its #2→gap1 (coverage/routing global), #6→gap2
(eval first-action), #7→native-mode framing (probe already built), #8→gap4 (GGUF), #10→gap5 (frontend).
Its "highest-value next fix" = **make the deterministic layer plugin-aware**, which subsumes A/C/B/E
and the gap-1 coverage gate. The plan (`quiet-singing-storm.md`) is restructured around that.

## Remediation plan (prioritized)

Legend — effort: S(<1h) / M(half-day) / L(multi-session). risk: behaviour change that needs a model
re-test vs. mechanical. **The plan was restructured around the plugin-aware-layer reframe after the
second review — see `C:\Users\admin\.claude\plans\quiet-singing-storm.md` for the current execution
plan; the table below is the first-review summary, kept for provenance.**

| # | Fix | Priority | Effort | Risk | Needs GPU? |
|---|---|---|---|---|---|
| 6 | Stop mislabeling non-chess tool errors | **P0** | S | mechanical | no |
| 1 | Scope chess routing/coverage to chess context | **P0** | M | behaviour (restraint!) | no (unit-testable) |
| 3 | Expand the STRESS suite to n≥60 | **P1** | S–M | authoring | no |
| 2 | Add a completion-grading eval tier | **P1** | M–L | new metric | yes (to run) |
| 4 | Align GGUF decode with HF (penalties off) | **P1** | S | fidelity re-test | yes (to verify) |
| 5 | Fix frontend reconcile (FEN fields, promo) | **P2** | M | UI-only | no |

### P0 — correctness, no model risk

**#6 (mechanical, do first):** In `tools.py::execute`, keep `chess.engine.EngineError` (+ timeouts)
→ `engine_unavailable`, but change the bare `except Exception` to a neutral `error: tool_failed
'<name>'` (name included so the model can re-route, and the user isn't told Stockfish is down for a
recipe-scaler bug). Add a test: a plugin tool that raises → `tool_failed`, not `engine_unavailable`.

**#1 (the important one — respects the deterministic-routing-restraint lesson):** The chess
deterministic layer must not fire outside chess. Two-part gate, NO new keyword regex (memory
`deterministic-routing-restraint` / `flexible-model-vs-deterministic-layers` warn against expanding
the matcher):
  (a) **Manifest intersection:** `matched_calls` / `routing_hints` only return a tool that is actually
      in the live `serving_tool_manifest(plugin_context)` — a force-routed tool must be callable.
  (b) **Domain gate:** apply the chess routing/coverage block in `respond` only when `chess-official`
      is enabled in `plugin_context`. When only an OOD bundle (e.g. life-skills) is active, the chess
      layer is a no-op → the model routes on its own (it's at 96.4%, it doesn't need the crutch).
  Residual (documented, not fixed by regex): a genuinely OOD prompt *inside* a chess-enabled serve
  could still match — but (a)+(b) kill the cross-domain case the reviewer flagged, and the model is
  the decider. Add unit tests: a tax prompt with life-skills-only context → empty `matched_calls`;
  with chess context → eval still fires (no regression on chess).

### P1 — strengthens the "model is good" proof + serve fidelity

**#3:** Add ~40–60 more held-out STRESS rows to `bench_suites.py` across more unseen domains
(finance, fitness, travel, coding-help, etc.), keeping unambiguous gold. Pure authoring, no risk;
directly answers "n=20 is smoke-level."

**#2:** New eval tier (own module, e.g. `eval_completion.py`) that runs the FULL `CoachLoop` per row
and scores: (i) args parse + validate, (ii) executor returned non-error, (iii) final reply cites the
tool result (reuse `_correct_eval_number`-style grounding check). This converts "routing accuracy"
into "task-completion accuracy" — the metric the reviewer (rightly) says is missing, and the one that
best proves the product works end-to-end. Reuses `bench_transcript`'s loop harness.

**#4:** Set GGUF decode to match HF (`repeat_penalty=1.0`, `top_p=1.0` at temp 0) in both
`model_gguf` methods. MUST re-test against the number-consistency guard (memory
`gguf-q4-fabricates-eval`) before shipping — penalties were possibly added to curb repetition, so
verify name-copying improves without a repetition regression. Gate behind a GPU smoke check.

### P2 — demo correctness (no model/eval impact)

**#5:** `sameBoard` → compare FEN fields [0..3] (placement, turn, castling, ep). `tryApplyUci` → pass
`uci[4]` as the promotion piece into `makeMove`. `makeMove` → honor a `promo` arg instead of forcing
queen. Frontend-only; add a board-reconcile test if a JS test harness exists, else manual repro.

## Next
Recommend executing **P0 (#6 then #1)** first — pure correctness, unit-testable, no GPU. Then P1 #3
(cheap proof win) and #2 (the completion metric). #4 and #5 batched when a GPU/browser pass is
scheduled. Await go-ahead on #1 and #4 (behaviour-affecting).
