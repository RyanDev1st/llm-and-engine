Parent: ../../harness-architecture.md

# Reliable multi-tool via deterministic coverage + budget forcing

How the chess-coach agent is made to use multiple tools **reliably in one session** —
the core objective — grounded in published test-time-scaling work, on a small
local model with no extra training.

## Status

Implemented + tested (2026-06-12). Default ON. Supersedes the staged Controller+
Narrator design (`2026-06-12-thinking-harness-design.md`), which was built, measured
live, and found to *underperform* the proven single loop (see Evidence).

## Problem (stated precisely)

The model is **capable** of multi-tool use — the live smoke showed it calling
`best_move` *and* `eval` in one turn. It just doesn't **reliably decide** to: on a
compound request ("3 best moves *and* the eval") it often calls one tool, or none,
then answers. That is a **reliability gap, not a capability gap** — and reliability
gaps are closed by deterministic constraints, not by hoping the model reasons.

## What the production reasoning systems actually do (grounded)

We surveyed how the strong "thinking" models work, to borrow the *mechanism*, not the
training cost:

- **s1 — budget forcing** ([arXiv 2501.19393](https://arxiv.org/abs/2501.19393),
  [simplescaling.github.io](https://simplescaling.github.io/)): control test-time
  compute by (1) **forcing an end-of-thinking token + "Final Answer:"** when over
  budget — *the "the user is waiting, answer now" behaviour* — and (2) appending
  **"Wait"** when the model stops too early, which makes it reconsider and often fix
  itself.
- **Anthropic — interleaved thinking**
  ([docs.claude.com](https://docs.claude.com/en/docs/build-with-claude/extended-thinking)):
  one continuous stream that thinks *between* tool calls (think → tool → reflect →
  next tool) under a token budget. Not a rigid two-pass split.
- **Gemini 2.5 — dynamic thinking budget**
  ([ai.google.dev](https://ai.google.dev/gemini-api/docs/thinking)): a `thinking_budget`
  cap; `-1` lets the model scale effort to perceived complexity.
- **DeepSeek-R1** ([arXiv 2501.12948](https://arxiv.org/abs/2501.12948),
  [Nature 2025](https://www.nature.com/articles/s41586-025-09422-z)): the gradual
  reasoning (self-reflection, verification) is **emergent from RL** — it is *trained*,
  not prompted.

**Honest framing (no fabrication):** serve-time tricks (s1/Gemini/Anthropic) only
*unlock latent* reasoning a model already has. R1's intelligence came from RL. Our
E2B is small and not reasoning-trained, so at serve time we raise **reliability**,
not the intelligence ceiling. Genuine reasoning lift = a future R1-style training run.

## Design — coverage layer on the single loop

We keep the proven, fast single loop (decide → tool → decide → reply) and add a
**deterministic coverage set** as the reliability constraint, with s1's mechanisms:

```
required = matched_tools(message)            # detected intents, e.g. {best_move, eval}
loop (cap 8):
  model decides: a <tool> call, or a final reply
  on a tool call:  execute; record the tool name (coverage) + full call (dedup)
  on a final reply:
     outstanding = required - gathered
     if none:                      return the reply            # all intents covered
     if intent not yet nudged:     inject "Wait — you still need {tool}, call it now"   # s1 "Wait"
     else:                         force-route {tool}          # backstop: guarantee
after cap:  inject "you're out of steps, the user is waiting — answer now"  # s1 forced-termination
```

- **The "Wait" steer is model-driven first** — it usually complies and *looks* like it
  reasoned its way to the second tool. The **backstop force-route guarantees** the tool
  runs if it doesn't. Completeness is never left to chance.
- **One continuous stream** (Anthropic's interleaved shape), not a two-pass split → no
  separate narrator call → fast, no fact-conflation.
- `required` is a **floor**: the model may gather more; on a greeting/out-of-scope it's
  empty and the model's first reply stands.
- **Dedup is by full call**, coverage by tool name — so `best_move depth=1` and
  `best_move top=3` both run (the staged design wrongly blocked the second).

`matched_tools` reuses the existing routing-hint matcher (`tool_hints.py`), so the same
keyword detection feeds both the prompt hint and the coverage guarantee — one source of
truth.

## Why staged was rejected (Evidence)

Live, on the real adapter, same prompt "give me the best move and the evaluation":

| Engine | Tools | Result |
|---|---|---|
| single (proven) | `best_move top=3` | **3 moves + evals** — correct |
| staged (Controller+Narrator) | `best_move depth=1`, `eval` | 1 shallow move + *guessed* alternatives; narrator conflated "+0.60 / 0.00 equal" |

Staged added a second model call per turn, biased the model toward lazy calls
(`depth=1`), and conflated facts — i.e. it was slower *and* worse than the loop it was
meant to improve. Retired to `legacy [ignore]/backend_thinking/`.

## Demo / ablation (for evaluation)

The web **Compare** toggle runs the same prompt with the coverage layer **ON vs OFF**,
side by side (board-isolated), so the before/after is visible: OFF whiffs on the
compound request; ON gathers every required tool. Clean ablation for a results table.

## Files

`backend/inference.py` (`CoachLoop.respond(coverage=True)`), `backend/tool_hints.py`
(`matched_tools`/`matched_calls`), `backend/web_app.py` (`coverage` variant),
`backend/server.py` (`coverage` flag), `gemma_chat_site/static/index.html` (Compare
toggle). Tests: `backend/test_coverage.py` (Wait-steer, backstop, ablation-off,
game-over skip, dedup-by-full-call) — 49 backend tests pass.

## Future (the real ceiling)

Genuine "the model thinks" — gradual, emergent reasoning about *which* tools — is an
**R1-style reasoning SFT/RL** step on E4B, deferred. The coverage layer here is the
reliable serve-time floor that holds in the meantime and carries over unchanged.
