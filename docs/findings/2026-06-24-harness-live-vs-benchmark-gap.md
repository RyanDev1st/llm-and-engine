Parent: [reference/harness-system-overview.md](../reference/harness-system-overview.md)

# Why live ≠ the 96% benchmark — the harness investigation

**Status:** Root-caused (static evidence, no GPU). Two serve-time parity flags shipped to A/B the
fixes (`CHESS_THIN_HARNESS`, `CHESS_BOARD_HOOK`); the GPU A/B + default flip is the next step.
**Scope:** the user's report — "Kaggle benchmark showed 96% skill-following and the prompts/replies
read fine; live runs are nowhere near. Either the harness forces the model to stop too soon, or the
model genuinely struggles in our harness." Both hypotheses tested against the code + corpus.

## Headline

**The 96% and the live serve are not the same prompt and not the same task.** The benchmark scores
a prompt the server never sends. So the 96% is real — but it does not predict live.

There are **three different prompts** in play:

| | Builder | Reasoning line | LIVE BOARD line | Turns | What it scores |
|---|---|---|---|---|---|
| **Train** | `build_system(catalog, reasoning_mode=MODE)` (loader, `data_pipeline`) | **yes** | **no** | full chains | next-token loss |
| **Benchmark** (`eval_routing.py`) | `build_system(catalog)` — **no mode** | **no** | **no** | first `<tool>`, single turn, 48 tokens |
| **Live** (`web_app.py:284` → `build_system_prompt`) | `build_system(LIVE catalog, MODE)` **+ prompt_start hook + memory** | **yes** | **yes** | full multi-step loop, multi-turn, + rescue layer |

Construction-level evidence (probe, both prompts built locally): live = ~1225 tok, benchmark row =
~1001 tok; the live prompt carries a `Reasoning mode: AUTO …` block, a `LIVE BOARD (current
position): …fen=…` line, and a fixed all-chess catalog, none of which the benchmark prompt has.

## Confirmed divergence #1 — the `LIVE BOARD` injection is off-distribution

The served system prompt appends, every turn:

```
LIVE BOARD (current position): turn=white, last_move=none, check=no, legal_moves=20, game_over=no, fen=…
```

via `chess_official.prompt_start` (`plugins/chess_official.py:45`), wired in
`build_system_prompt` (`inference.py:644`). **Training never had it:** `0 / 2731` val rows have a
system message containing `LIVE BOARD` / `Reasoning mode` / `current position`; rows store no system
message at all — the loader rebuilds it from `build_system(...)`, which `data_pipeline` confirms does
**not** reference `prompt_start` or any board line.

This directly **contradicts the trained contract.** The `chess-coach` skill body the model learned
says: *"The board is live but **hidden** — call `board_state` before asserting turn, FEN, last move…
Never claim a board fact from memory."* At serve the board is **not** hidden — it is in the system
prompt. The model is being fed a prompt shape it never saw, that contradicts its own loaded skill.
Plausible live symptoms: narrating board facts ("position is equal at move 0") straight off the
injected `fen=startpos` line instead of routing to a tool; or wasted/confused first steps.

→ Shipped `CHESS_BOARD_HOOK` (default ON = current behavior; `=0` drops the line so live == trained).

## Confirmed divergence #2 — the benchmark scores an easier, narrower task

`eval_routing.first_turn` (`eval_routing.py:44`) feeds `[system(bare), one user turn]`, generates
**48 tokens**, stops at `</tool>`, and scores **only the first tool**. That is essentially *fast-mode,
single-turn, first-action* routing. Live runs **auto mode** (UI default) — the model emits
`<goal>` + interleaved `<think>` + an action, across **multiple steps and multiple turns**, then the
full rescue layer post-processes it. **The 96% never measured the auto-mode, multi-step, multi-turn
behavior that live actually uses.** (The corpus *did* train auto/think with `<think>`/`<goal>`, so the
mode itself is in-distribution — it's the benchmark that omits it, not the serve.)

The faithful tier already exists: `eval_completion.py` runs the **full CoachLoop** per row
(`eval_completion.py:135`) through `build_system_prompt` (so it includes the LIVE BOARD hook + mode +
rescue layer). It has simply never been the headline number — and per the remediation plan it was
staged for the GPU pass, not yet run as the live-faithful metric.

## The "stop too soon" hypothesis — partially true, ranked

Real loop paths that can end a turn before the model is "done":
1. **The rescue layer substitutes the answer.** `_force_answer` / `_verify_fulfilled` /
   `_force_synthesis` (`inference.py:660`, `:683`, `:705`) inject a hidden generation and can replace
   the model's continued reasoning with a forced reply — off-distribution nudges the model never
   trained on. (S1 disables these.)
2. **`decision is None` finalizes on any tagless prose** (`inference.py:~903`): if an auto-mode step
   emits reasoning prose without closing a `<tool>`/`<skill>` tag, it is treated as the final reply.
3. **Per-step token cap**: `step_cap = 224` in fast/"" mode, `320` in auto/think (`inference.py:861`).
   A long `<goal>`+`<think>`+action *usually* fits, and a cut-off `<tool>` is recovered — so this is a
   secondary risk, not the main one.

But note: the model was **trained to do ONE action then narrate** (Mode 1 → Mode 2; the benchmark even
checks "no second `<tool>` after a result"). So single-tool turns are **by design**, not a bug. The
"stops too soon / flaky" *feeling* in live is dominated by divergence #1/#2 and the multi-turn
deflection on terse input (separate skill-body fix, commit `d7aa1d17`), not by the loop truncating.

## Verdict on the two hypotheses

- **"The model genuinely struggles in our harness" — TRUE and primary.** The live prompt is
  off-distribution (the `LIVE BOARD` injection; confirmed absent from all training), and live exercises
  auto-mode multi-step multi-turn that the 96% never measured.
- **"The harness forces it to stop too soon" — TRUE but secondary.** The rescue layer's hidden
  generations + the tagless-prose finalize can short-circuit a turn; these are the off-distribution
  add-ons S1 turns off. They degrade rather than truncate.
- **Plus a measurement artifact:** the benchmark grades a prompt the server doesn't send, so "96%"
  over-states live capability. The fix is to make `eval_completion` (live-faithful) the headline.

This matches the user's own recollection that the harness "ran fine before the `<think>`/rescue
changes": the regression is in the serve **layer on top of the model**, not the weights — exactly the
recurring lesson (`flexible-model-vs-deterministic-layers`, `chess-routing-fires-on-ood`).

## Recommendation — the A/B that decides it (GPU, measured)

Run `eval_completion` (full loop, auto mode, the OOD STRESS + a chess slice) as the headline, across
the 2×2 of the new flags, plus a live multi-turn pass:

| Run | `CHESS_BOARD_HOOK` | `CHESS_THIN_HARNESS` | Tests |
|---|---|---|---|
| Current default | 1 | 0 | baseline (what users see now) |
| Parity only | **0** | 0 | does removing the off-distribution board line alone recover it? |
| Thin only | 1 | **1** | does dropping the rescue layer alone recover it? |
| Both | **0** | **1** | the full "trust the trained model" config |

Hypothesis from this analysis: **board-hook OFF is the single biggest win** (restores prompt parity),
**thin ON** removes the remaining off-distribution nudges + latency, and **both** is closest to the
distribution the 96% was measured on. Flip the defaults to whichever wins; keep the narrow 4B guards
(number-fabrication, reload nudge, format recovery) in all configs. No change to the contract, verbs,
modes, or corpus.

## Artifacts shipped this session

- `d7aa1d17` chess-coach skill body: meta/capability route + act-on-terse-replies (the immediate flaky
  symptoms).
- `f4cbce5e` `CHESS_THIN_HARNESS` (S1) — gate off coverage force-routing + self-verify probe + ask-back
  re-gen; default off; scripted-model tests.
- `86f7bc28` `CHESS_BOARD_HOOK` — gate the `LIVE BOARD` injection; default on; parity test.
- This finding + `2026-06-24-harness-vs-claude-code-codex.md` (the principle-level side-by-side).
