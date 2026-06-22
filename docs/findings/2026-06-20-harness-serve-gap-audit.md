Parent: docs/reference/harness-architecture.md

# Harness serve-loop gap audit (2026-06-20)

## Status
Report only — **no fixes applied** (user elected report-first). Two gaps PROVEN from the
running `CoachLoop`; one known larger feature gap. Tier 1 (`</skill>` stop) and Tier 3
(name/arg validation) from the same hardening pass are already shipped (commits 5c1ae0a6,
acd1e67d) and are NOT repeated here.

## Scope
Re-audit of the live LLM serve loop (`src/llm/backend/inference.py` `CoachLoop` +
`tools.py` executor), grounded against proven agent harnesses (Anthropic documented
agentic tool-use loop + extended-thinking block separation; OpenAI function-calling +
reasoning items). Goal: find REAL gaps, not fabricated ones — every claim below is
verified against code or a runnable repro.

## Evidence

### Matches proven patterns (verified clean — not gaps)
- Loop shape generate → execute action → feed `tool_result` → repeat until a no-action
  final == Anthropic `tool_use → tool_result → … → end_turn`.
- Iteration cap + graceful finalize (`MAX_TOOL_CALLS=8` then budget-forced answer).
- Tool errors fed back as corrective `error:` strings for self-correction.
- Duplicate-call dedup; context-window eviction (`window.fit`); sandbox output/time caps
  (`sandbox.py`); stop tokens (incl. `</skill>` after Tier 1); tool-tag leak guards.
- One action per generation step — intentional divergence from parallel tool calls;
  correct for a small model.

### G1 — `<think>` (and stray `<goal>`) leak into the visible final reply  [PROVEN]
Contract (`renderer/thinking.py:14`): *"Serve strips `<think>` from the visible reply (the
'thinking' panel) and `<goal>` to the plan panel."* `gated_answer` emits `<think>…</think>`
before think/auto finals. The frontend thinking-panel (`gemma_chat_site/static/index.html`)
is built from **tool events**, not by parsing `<think>` out of the reply text, and
`CoachLoop._finalize` does not strip it.

Repro (coverage off to isolate the final-reply path):
```
M = ScriptedModel(['<think>I have what I need - answer now</think> You are slightly better; develop your knight.'])
out = CoachLoop(M, ToolExecutor(Game(), None)).respond([], 'give me general advice', coverage=False)
# out['reply'] == '<think>I have what I need - answer now</think> You are slightly better; ...'
# '<think>' in out['reply'] -> True   (LEAK)
```
This is exactly the separation Anthropic/OpenAI enforce at the protocol level (reasoning
block ≠ visible `text`). Surfaces whenever reasoning mode is actually used at serve.

Fix direction: strip `<think>…</think>` and stray `<goal>…</goal>` from the reply in
`_finalize` (backend, ~surgical). Live-stream routing of `<think>`→panel during streaming
is a smaller frontend follow-up (the token stream emits `<think>` tokens before the strip).

### G2 — direct-answer-with-`<goal>` mis-classified as a plan panel  [PROVEN]
`prepend_open_goal` prepends a leading `<goal>` to the FIRST trained assistant turn; for a
think/auto turn that answers DIRECTLY (no action, e.g. V1_Q), that first turn IS the final
reply. `is_plan_panel` treats any `<goal>`-bearing turn with no executable action as a
panel, so the answer is shunted to the plan panel and the loop re-generates — discarding
the answer text.

Repro:
```
raw = '<goal>tell them what is up</goal>\n<think>no skill needed - answer plainly</think> It is your move; you are fine.'
is_plan_panel(raw)  # -> True   (should be False: bare <goal> + prose answer, no <plan> checklist)
```
Fix direction: `is_plan_panel` should require a `<plan>` checklist (or no real prose after
the tags), so a goal-prefixed direct answer is not treated as a panel.

### G3 — reasoning mode not wired at serve  [known, larger]
`build_system_prompt(reasoning_mode="")` is hardcoded and `CoachLoop.respond` has no mode
param (code comment at `inference.py` flags this as deferred). The trained fast/think/auto/
plan toggle is not selectable at serve. Feature gap, not a correctness bug; threading a mode
through respond → web_app/server → UI is the larger change. Note: fixing G1 first is a
prerequisite so think/auto don't leak once the toggle is live.

## Next
Decision pending. Recommended order when greenlit: G1 + G2 together (~15 lines + regression
tests, frozen-model/harness-only posture), then G3 as a separate feature. See memory
[[harness-serve-hardening]] for the durable lessons.
