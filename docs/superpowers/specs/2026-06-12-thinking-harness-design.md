Parent: ../../harness-architecture.md

# Thinking harness — design spec

A serve-time staged reasoning loop for the chess-coach agent. It forces the model
to **decide what to do and what not to do** in dedicated, separated stages — and
above all to **fact-check that the user's request is actually fulfilled** before
replying — instead of running a single reflexive tool loop. No retraining.

## Status

Design approved (2026-06-12). Ready for an implementation plan.

## Problem

Today the agent runs a single-prompt loop (`backend/inference.py:CoachLoop.respond`):
each step is an optional **canned lead-in** ("Now the engine's read.") + one
`<tool>` call, looping until the model emits a plain reply or hits the 6-call cap.
The lead-in is narrative theater (`llm_dataset/v1/renderer/leadins.py`) — it does
not reason about whether a tool is needed, which tool, or whether the goal is met.
The deterministic layer (`tool_hints.py`) nudges tool choice from outside, but the
model never explicitly checks "is the request fulfilled?" This yields the observed
mis-routes: stopping one tool short, over-calling, narrating before the goal is
reached, and answering out-of-scope requests with a tool call.

A staged Router/Narrator split was prototyped earlier in `src/llm/llm_runtime/`
(JSON output schemas + payload contracts + mode invariants) but stranded
(`src/llm/README.md:19`) in favor of the text `<tool>` format. This design revives
the *staged* idea, leaner and richer: it keeps the proven `<tool>` text format,
adds a **Verifier** stage that fact-checks the goal after every tool, and scopes
each stage's context so the model focuses on one decision at a time.

## Goals / non-goals

**Goals**
- Force an explicit, separated decision at each step: route → act → **verify goal** → (loop or narrate).
- Decide *on the fly* when more than one tool is needed (multi-tool emerges from the verify→route loop, not upfront planning).
- Each stage is its own dedicated, robust unit: one system prompt, one job, scoped inputs.
- Serve-time only. No new training, no corpus changes, no model dependency. Works on the current E2B adapter; benefits E4B later for free.
- Never regress: the proven single-prompt loop stays available behind a toggle.

**Non-goals**
- No training/SFT changes (revisit "bake the stages into the corpus" only after live validation, likely with E4B).
- No new tools, skills, or plugin changes. The 12-tool manifest and skill model are unchanged.
- No JSON output contract (format C keeps `<tool>` text + keyword verdicts; the `llm_runtime` JSON path stays stranded).

## Decisions (locked)

| Decision | Choice |
|---|---|
| Realization | **Serve-time only**, no retrain. Each stage sees only the context it needs, not the full history. |
| Stages | **Router + Verifier (every step) + Narrator.** Verifier owns the stop decision. |
| Control format | **C (hybrid):** tool calls as `<tool>NAME args</tool>` (reuse `extract_call`/`parse_call`); control verdicts as single keywords (`REPLY:`, `DONE`, `MORE:`). |
| Rollout | Env toggle `CHESS_THINKING=staged|single`, **default `single`** (proven) until live-validated, then flip. |

## Architecture

The staged loop replaces the inner loop of `CoachLoop.respond` when
`CHESS_THINKING=staged`; otherwise the existing single-prompt loop runs unchanged.

```
respond(history, user_message):
  goal  = user_message            # the request to satisfy this turn
  facts = []                      # tool results gathered THIS turn (compact)
  for step in 1..MAX_STEPS(6):
     route = ROUTER(goal, facts_summary, board_facts, tools+hints)   # dedicated prompt #1
     if route is REPLY:  return route.text          # no tool needed (greeting / out-of-scope / already known)
     if route.tool seen before (dedup): break       # Router repeated a tool -> stop, narrate what we have
     result = execute(route.tool)                   # deterministic; extract_call recovery reused
     facts.append(result)
     verdict = VERIFIER(goal, facts_summary, result)                 # dedicated prompt #2
     if verdict is DONE: break
     # CONTINUE(missing): verdict.missing is scoped into the next ROUTER call
  return NARRATOR(goal, facts_summary)                               # dedicated prompt #3
```

- **Multi-tool on the fly:** Verifier emits `MORE: need the opponent's threats too` →
  the next Router call routes `threats`. The model never plans the whole sequence
  upfront; it grows the tool chain only as the Verifier finds gaps.
- **Separation of concerns:** Router = "what next", Verifier = "are we done", Narrator
  = "say it". None can do another's job; each is independently testable.

### The three stages

| Stage | Job | Scoped payload (sees) | Emits |
|---|---|---|---|
| **Router** | pick the next tool, or reply directly | goal · `facts_summary` · cheap board facts (turn / legal-count / last-move / check) · tool+skill manifest · deterministic hints | one `<tool>…</tool>` **or** `REPLY: <text>` |
| **Verifier** | fact-check: is the goal reached? | goal · `facts_summary` · the **latest** tool result | `DONE` **or** `MORE: <missing fact>` |
| **Narrator** | write the grounded user reply | goal · `facts_summary` (compact) | plain text (+ guiding question) |

### Dedicated system prompts (terse, one job each)

- **Router:** "You are the router. Output EXACTLY one next action: a single
  `<tool>NAME arg=value</tool>` to gather a fact or act, OR `REPLY: <answer>` if no
  tool is needed (greeting, out-of-scope, or the goal is already answerable from the
  facts). Call only listed tools while their applies_when holds. Do not narrate."
- **Verifier:** "You are the verifier. Given the user's goal and the latest tool
  result, decide if the request can now be fully answered. Output EXACTLY `DONE`, or
  `MORE: <the one specific fact still missing>`. Do not call tools. Do not narrate."
- **Narrator:** "You are the narrator. Using ONLY the gathered facts, write a short
  grounded reply. Never invent numbers (positive score = white better). End a
  coaching answer with one brief guiding question. No tool tags."

Each is assembled with the harness contract header it needs (Router needs the tool
manifest + skills catalog + plugins; Verifier and Narrator do not), via the
existing `build_system()` plus a stage header.

### Scoped context (the focus principle — built deterministically, no extra model calls)

- `facts_summary`: a compact `tool→key-result` line list with lead-ins stripped,
  e.g. `eval→+0.30 pawns; threats→none significant`. Derived from the turn's
  `tool_calls`/`tool_results`, not the raw transcript.
- **Board facts** for the Router are read straight from `game` (turn, legal-count,
  last-move, check) so the Router knows the situation without spending a
  `board_state` tool step.
- Prior turns: only a short rolling summary (or the last assistant reply) is carried,
  never the full multi-turn transcript. `window.fit` still bounds the total.

## Control format (C) and parsing

- Tool calls: the existing `<tool>NAME args</tool>`, parsed by `toolfmt.parse_call`,
  recovered by `inference.extract_call` (handles `<tool_code>`, malformed wrappers,
  hint-echo, stop-trimmed close tags).
- Router direct reply: line starting `REPLY:` — the rest is the user-facing text.
- Verifier verdict: `DONE` (exact, case-insensitive) or `MORE: <text>`.
- A thin `thinking/parse.py` exposes `parse_router(raw) -> ToolAction|ReplyAction`
  and `parse_verifier(raw) -> Done|More(missing)`.

## Error handling / caps (fail toward answering, never spin)

| Condition | Behavior |
|---|---|
| `MAX_STEPS` (6) exceeded | force Narrator on the facts gathered so far |
| Verifier verdict unparseable (not DONE/MORE) | treat as **DONE** (stop + narrate) — avoids infinite loop |
| Router routes a tool already run this turn (dedup) | break to Narrator |
| Router output is neither a tool nor `REPLY:` | `extract_call` recovery first; else treat its text as the reply |
| Narrator empty or leaks a tag | existing `_fallback_reply` / `narrate_tool_result` backstop |
| game over | Router gets the game-over hint; Verifier returns `DONE` fast |

## Files (feature folder; each source file < 200 lines per repo cap)

```
src/llm/backend/thinking/
  __init__.py
  prompts.py   ROUTER / VERIFIER / NARRATOR system prompts + scoped-payload builders
  parse.py     parse_router (reuses extract_call) + parse_verifier
  loop.py      StagedLoop — orchestrates router→execute→verifier→narrate with caps/fallbacks
```

Integration: `CoachLoop.respond` reads `CHESS_THINKING`; when `staged`, it delegates
the inner loop to `StagedLoop` (constructed with the same `model`, `executor`,
`agent_overlay`, `plugin_context`, and the deterministic hint functions). The public
`respond` return shape (reply, tool_call(s), tool_result(s), turns, context) is
unchanged so `web_app.py` / `server.py` need no changes. Tools, skills, plugins, and
the deterministic layer are reused as-is.

## Testing

Scripted-model stage tests (extend the `ScriptedModel` pattern in
`backend/test_serve_smoke.py`), each asserting stage order + stop behavior:

1. **Full path:** Router→`<tool>eval>`; Verifier→`MORE: need threats`;
   Router→`<tool>threats>`; Verifier→`DONE`; Narrator→reply. Assert both tools ran in
   order, loop stopped on DONE, reply returned, no leak.
2. **Immediate reply:** Router→`REPLY: hi there` → returns at once, zero tools.
3. **Malformed verdict:** Verifier→"maybe" → treated DONE, narrates (no spin).
4. **Cap:** Verifier always `MORE` → stops at MAX_STEPS, forces Narrator.
5. **Dedup:** Router repeats the same tool → breaks to Narrator.
6. **Game over:** finished board → Verifier `DONE` fast, no analysis tool.

Then a live in-process smoke (real adapter) comparing `single` vs `staged` on a few
turns (eval, multi-tool "how am I doing and any threats?", out-of-scope, game over),
confirming grounded replies and no leaks.

## Rollout

- Ship behind `CHESS_THINKING`, default `single`. Validate `staged` live (smoke +
  the dual-mode demo can show thinking on/off). Flip the default to `staged` once it
  matches or beats `single` on the mis-route cases. Single stays as the fallback.

## Future (out of scope here)

- Bake the staged reasoning into the SFT corpus (trained `<think>`/role-scoped data)
  only after serve-time validation — likely alongside the E4B decision, since a
  richer reasoning step may be E4B-class.
- A number-consistency guard in the Narrator (verify stated numbers against facts)
  and magnitude grounding — deterministic companions to the Verifier.
