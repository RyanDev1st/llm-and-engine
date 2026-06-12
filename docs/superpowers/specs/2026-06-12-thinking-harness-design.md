Parent: ../../harness-architecture.md

> **SUPERSEDED (2026-06-12)** by [`2026-06-12-coverage-reliability-design.md`](2026-06-12-coverage-reliability-design.md).
> This staged Controller+Narrator design was built and measured live — it was slower
> *and* produced worse tool calls than the proven single loop (see that doc's Evidence).
> The reliability win (deterministic multi-tool coverage) was kept and moved onto the
> single loop with s1-style budget forcing; the staged code is retired to
> `legacy [ignore]/backend_thinking/`. Kept here as the design journey.

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
the *staged* idea, leaner: it keeps the proven `<tool>` text format, makes the
per-step decision **explicitly fact-check the goal** before routing, and scopes
each stage's context so the model focuses on one decision at a time.

## Goals / non-goals

**Goals**
- Force an explicit, fact-checking decision at each step: *is the goal reached?* → stop and narrate, else route the next tool.
- Decide *on the fly* when more than one tool is needed, and **guarantee coverage** of compound/multi-part requests via a deterministic `required` set (the loop won't narrate while a detected intent is ungathered).
- Dedicated, robust stages: each has its own system prompt, one job, scoped inputs.
- **One model call per step** (perf-safe: never two concurrent/back-to-back model calls for a single tool).
- Serve-time only. No new training, no corpus changes. Works on the current E2B adapter; benefits E4B later for free.
- Web UI: see the difference between staged and single, and toggle thinking mode on/off.
- Never regress: the proven single-prompt loop stays available behind a toggle.

**Non-goals**
- No training/SFT changes (revisit "bake the stages into the corpus" only after live validation, likely with E4B).
- No new tools, skills, or plugin changes. The 12-tool manifest and skill model are unchanged.
- No JSON output contract (format C keeps `<tool>` text + keyword control; the `llm_runtime` JSON path stays stranded).

## Decisions (locked)

| Decision | Choice |
|---|---|
| Realization | **Serve-time only**, no retrain. Each stage sees only the context it needs, not the full history. |
| Stages | **Controller + Narrator.** The Controller fuses verify + route into **one model call per step** — its forced first step is the goal fact-check. The Narrator writes the grounded reply. |
| Per-step cost | **One model call per step** (not two). Slower per call but safer for GPU load; verify and route share the call. |
| Multi-tool coverage | A deterministic `required = matched_tools(message)` set. The loop refuses `DONE` while a required tool is outstanding and force-routes it; the Controller may gather more, never less. `required` is a floor (empty → trust the Controller). |
| Control format | **C (hybrid):** the Controller emits a `<tool>NAME args</tool>` call (reuse `extract_call`/`parse_call`) or the single keyword `DONE`; the Narrator emits plain text. |
| Web toggles | A **thinking-mode toggle** (staged on/off) and a **compare toggle** (same prompt through staged + single, side by side). |
| Rollout | Env toggle `CHESS_THINKING=staged|single`, **default `single`** (proven) until live-validated, then flip. Web toggles override per request. |

## Architecture

The staged loop replaces the inner loop of `CoachLoop.respond` when thinking is on;
otherwise the existing single-prompt loop runs unchanged.

```
respond(history, user_message):
  goal     = user_message                  # the request to satisfy this turn
  required = matched_tools(user_message)    # deterministic coverage set, e.g. {best_move, eval}; may be empty
  facts    = []                             # tool results gathered THIS turn (compact)
  seen     = set()                          # tools already run this turn
  for step in 1..MAX_STEPS(10):
     outstanding = [t for t in required if t not in seen]   # required facts not yet gathered
     action = CONTROLLER(goal, facts_summary, board_facts, tools+hints, outstanding)   # ONE model call
     if action is DONE or action.tool in seen:
         if not outstanding: break          # fully covered (or nothing new to add) -> narrate
         action = force_tool(outstanding[0]) # backstop: gather a required-but-missing fact deterministically
     facts.append(execute(action.tool)); seen.add(action.tool)
  return NARRATOR(goal, facts_summary)                               # dedicated prompt #2
```

Flow:

```
 user message
   |
   v
 goal = message
 required = matched_tools(message)      # deterministic coverage set, e.g. {best_move, eval}
 facts = [] ;  seen = {}
   |
   v
 step = 1
   |
   v
 outstanding = required - seen          # required facts not yet gathered
   |
   v
+==================================================+
|  CONTROLLER  (ONE model call)                    |  <- dedicated prompt #1
|  sees: goal + facts_summary + board facts        |
|        + tool/skill manifest + hints             |
|        + OUTSTANDING (required, not yet gathered) |
|  rule: DONE only when EVERY part is covered      |
|  emits:   <tool>NAME args</tool>   |   DONE       |
+==================================================+
   |
   v
< DONE,  OR  tool already in seen? >
   |  yes                         \  no
   v                               \
< outstanding empty? >              \
   |  yes          |  no             v
   v               v            execute chosen tool
[NARRATOR]   force outstanding[0]   -> facts, seen
             (deterministic:        |
              gather missing fact)  |
             -> facts, seen         |
             |                      |
             +----------+-----------+
                        v
                 < step >= MAX_STEPS (10) ? >---- yes ----> [NARRATOR]
                        |  no
                        v
                   step++ ; loop back to CONTROLLER

+==================================================+
|  NARRATOR  (ONE model call)                      |  <- dedicated prompt #2
|  sees: goal + facts_summary  (no tools)          |
|  emits: grounded reply (+ guiding question)      |
|  backstop: empty/leak -> fallback narrate        |
+==================================================+
   |
   v
 reply to user
```

- **The fact-check is forced and free of an extra call:** every step the Controller
  must first answer "is the goal already satisfied?" — `DONE` if yes, else it routes
  the next tool. The goal-check is the `DONE` decision; no separate inference.
- **Guaranteed multi-tool coverage:** `required = matched_tools(message)` is the set of
  tools the deterministic hint layer maps the request to (it already detects multiple
  intents, e.g. "best move **and** evaluation" → `{best_move, eval}`). The loop refuses
  to narrate while any required tool is `outstanding` — a premature `DONE` (or a
  duplicate) force-routes the missing tool. The Controller may gather *more* than
  required, never *less*. Compound requests are guaranteed, not hoped for. (`required`
  is a floor; when the hint layer detects nothing, it is empty and the Controller's
  `DONE` is trusted — e.g. greetings.)
- **Separation:** Controller = "are we done, and if not what next"; Narrator = "say
  it, grounded, no routing." Two dedicated prompts, each independently testable.

### The two stages

| Stage | Job | Scoped payload (sees) | Emits | Calls |
|---|---|---|---|---|
| **Controller** | fact-check the goal, then route if needed | goal · `facts_summary` · cheap board facts (turn / legal-count / last-move / check) · tool+skill manifest · deterministic hints · `outstanding` (required tools not yet gathered) | `<tool>…</tool>` **or** `DONE` | 1 per step |
| **Narrator** | write the grounded user reply | goal · `facts_summary` (compact) | plain text (+ guiding question) | 1 per turn |

### Dedicated system prompts (terse, one job each)

- **Controller:** "You are the controller. FIRST decide: is EVERY part of the user's
  goal satisfied by the facts gathered so far? If yes, output EXACTLY `DONE`. If not
  (including any still-needed facts listed under OUTSTANDING), output the single next
  `<tool>NAME arg=value</tool>` that gets a missing fact or performs the action. Call
  only listed tools while their applies_when holds. Output ONLY `DONE` or one tool
  call — never narrate." (OUTSTANDING is injected when the deterministic coverage set
  has ungathered tools.)
- **Narrator:** "You are the narrator. Using ONLY the gathered facts, write a short
  grounded reply (if there are no facts, answer the user directly or decline if
  out-of-scope). Never invent numbers (positive score = white better). End a coaching
  answer with one brief guiding question. No tool tags."

The Controller is assembled with the full harness contract (tool manifest + skills
catalog + plugins, via `build_system()`) plus its stage header; the Narrator gets a
minimal header (no tool manifest — it cannot route).

### Scoped context (the focus principle — built deterministically, no extra model calls)

- `facts_summary`: a compact `tool→key-result` line list with lead-ins stripped,
  e.g. `eval→+0.30 pawns; threats→none significant`. Derived from the turn's
  tool calls/results, not the raw transcript.
- **Board facts** for the Controller are read straight from `game` (turn, legal-count,
  last-move, check) so it knows the situation without spending a `board_state` step.
- Prior turns: only a short rolling summary (or the last assistant reply), never the
  full multi-turn transcript. `window.fit` still bounds the total.

## Control format (C) and parsing

- Tool calls: the existing `<tool>NAME args</tool>`, parsed by `toolfmt.parse_call`,
  recovered by `inference.extract_call` (handles `<tool_code>`, malformed wrappers,
  hint-echo, stop-trimmed close tags).
- Controller stop signal: the single keyword `DONE` (exact, case-insensitive).
- Narrator: plain text, no tags.
- A thin `thinking/parse.py` exposes `parse_controller(raw) -> ToolAction | Done`.

## Error handling / caps (fail toward answering, never spin)

| Condition | Behavior |
|---|---|
| `MAX_STEPS` (10) exceeded | force the Narrator on the facts gathered so far |
| Controller output is neither a tool nor `DONE` | `extract_call` recovery first; if a tool is recovered, run it; else treat as `DONE` (then: outstanding → force-route it, else narrate) |
| `DONE` or a duplicate tool **while a required tool is outstanding** | do NOT stop — `force_tool(outstanding[0])` gathers the missing required fact deterministically |
| `DONE` or a duplicate tool with **nothing outstanding** | stop → Narrator |
| First step is `DONE` with no facts and empty `required` (greeting / out-of-scope) | Narrator replies from the goal alone |
| Narrator empty or leaks a tag | existing `_fallback_reply` / `narrate_tool_result` backstop |
| game over | Controller gets the game-over hint and returns `DONE` fast (no analysis) |

## Web UI

The server exposes the staged engine and the comparison so the difference is visible.

- **`/api/chat` gains `thinking: "staged" | "single"`** (default from `CHESS_THINKING`),
  alongside the existing `variant`. The response carries a `trace`: the ordered stage
  decisions (`controller: <tool>… | DONE`, then `narrator`) for the verbose panel.
- **Thinking-mode toggle:** staged ON → the chat runs the staged engine; OFF → the
  single-prompt loop. Sets `thinking` on each `/api/chat`.
- **Compare toggle:** ON → the same prompt is run through BOTH `staged` and `single`
  and rendered side by side (reusing the dual-panel layout), so the user sees how the
  two methods differ. Mutually exclusive with the existing SFT-vs-base dual mode (a
  small selector picks which comparison the two panels show).
- **Verbose stage trace:** the existing collapsible "thinking" panel surfaces the
  Controller's per-step decisions and the Narrator when staged is on — making the
  forced reasoning visible, not hidden.

## Files (feature folder; each source file < 200 lines per repo cap)

```
src/llm/backend/thinking/
  __init__.py
  prompts.py   CONTROLLER / NARRATOR system prompts + scoped-payload builders
  parse.py     parse_controller (reuses extract_call) -> tool action or DONE
  loop.py      StagedLoop — orchestrates controller→execute→…→narrate; coverage set, caps, fallbacks, trace
```

Supporting change in `tool_hints.py`: factor out `matched_tools(message) -> set[str]`
(the tool names the routing triggers match) so it feeds both the existing hint string
and the coverage `required` set — one source of truth, no duplicate regexes. `StagedLoop`
also needs a small `force_tool(name)` that builds a default call for a required-but-
missing tool (analysis tools → default depth; `move` → the SAN the hint extracted).

Integration: `CoachLoop.respond` reads the per-request `thinking` flag (defaulting to
`CHESS_THINKING`); when staged, it delegates the inner loop to `StagedLoop`
(constructed with the same `model`, `executor`, `agent_overlay`, `plugin_context`, and
the deterministic hint functions). The public `respond` return shape (reply,
tool_call(s), tool_result(s), turns, context) is preserved and extended with `trace`.
`web_app.chat` / `server.py` add the `thinking` flag and the compare path. Tools,
skills, plugins, and the deterministic layer are reused unchanged.

## Testing

Scripted-model stage tests (extend the `ScriptedModel` pattern in
`backend/test_serve_smoke.py`):

1. **One-tool path:** Controller→`<tool>eval>`; Controller→`DONE`; Narrator→reply.
   Assert the tool ran, loop stopped on DONE, reply returned, no leak, trace recorded.
2. **Multi-tool on the fly:** Controller→`<tool>eval>`; Controller→`<tool>threats>`;
   Controller→`DONE`; Narrator→reply. Assert both tools ran in order.
3. **Guaranteed coverage (the key test):** message "best move and the evaluation" →
   `required={best_move, eval}`. Controller→`<tool>best_move>` then **prematurely**
   `DONE`; assert the loop force-routes `eval` anyway (both gathered) before narrating.
4. **Immediate done (greeting/out-of-scope):** empty `required`, Controller→`DONE`
   with no facts → Narrator replies, zero tools.
5. **Malformed Controller:** unparseable output → recovered tool, else DONE (no spin).
6. **Cap:** Controller always routes → stops at MAX_STEPS, forces Narrator.
7. **Dedup with nothing outstanding:** Controller repeats a tool → stops to Narrator.
8. **Game over:** finished board → Controller `DONE` fast, no analysis tool.
9. **Compare path:** `web_app.chat(..., thinking)` and the compare mode return both
   staged and single outputs for the same prompt.

Then a live in-process smoke (real adapter) comparing `single` vs `staged` on a few
turns (eval, multi-tool "how am I doing and any threats?", out-of-scope, game over),
confirming grounded replies and no leaks.

## Rollout

- Ship behind `CHESS_THINKING`, default `single`. Validate `staged` live (the compare
  toggle makes the difference visible). Flip the default to `staged` once it matches
  or beats `single` on the mis-route cases. Single stays the fallback.

## Future (out of scope here)

- Bake the staged reasoning into the SFT corpus (trained reasoning data) only after
  serve-time validation — likely alongside the E4B decision, since a richer reasoning
  step may be E4B-class.
- A number-consistency guard in the Narrator (verify stated numbers against facts)
  and magnitude grounding — deterministic companions to the Controller's fact-check.
