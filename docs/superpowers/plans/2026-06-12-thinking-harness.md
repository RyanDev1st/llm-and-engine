# Thinking Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a serve-time staged reasoning loop (Controller + Narrator) that forces the model to fact-check goal-completion each step and guarantees multi-tool coverage, behind `CHESS_THINKING` (default `single`), with web toggles to compare it against the proven single-prompt loop.

**Architecture:** A new `backend/thinking/` package. Each turn loops a single **Controller** model call (verify goal → emit one `<tool>` or `DONE`) until done; a deterministic `required` coverage set (from the existing hint layer) force-routes any missing intent; then one **Narrator** call writes the grounded reply. `CoachLoop.respond` delegates to `StagedLoop` when thinking is on. No retraining; tools/skills/plugins/deterministic-layer reused unchanged.

**Tech Stack:** Python 3.13, python-chess, stdlib `http.server`, pytest. Frontend: vanilla JS in `gemma_chat_site/static/index.html`.

**Spec:** `docs/superpowers/specs/2026-06-12-thinking-harness-design.md`

---

## File Structure

| File | Responsibility | New/Modify |
|---|---|---|
| `src/llm/backend/tool_hints.py` | add `matched_calls`/`matched_tools` (coverage set) by factoring out the matcher | Modify |
| `src/llm/backend/thinking/__init__.py` | package marker | Create |
| `src/llm/backend/thinking/prompts.py` | Controller/Narrator system prompts + scoped-payload builders (`board_facts`, `facts_summary`, user-message builders) | Create |
| `src/llm/backend/thinking/parse.py` | `parse_controller(raw)` → `("tool", call)` or `("done", None)` | Create |
| `src/llm/backend/thinking/loop.py` | `StagedLoop` orchestrator: coverage, caps, force-route, trace | Create |
| `src/llm/backend/inference.py` | `CoachLoop.respond(..., thinking=None)` delegates to `StagedLoop` when staged | Modify |
| `src/llm/backend/web_app.py` | `chat(..., thinking)`, `"thinking"` compare variant, `loop_mirror` | Modify |
| `src/llm/backend/server.py` | pass `thinking` from the request body | Modify |
| `src/llm/gemma_chat_site/static/index.html` | thinking-mode toggle + compare toggle + send wiring | Modify |
| `src/llm/backend/thinking/test_staged_loop.py` | scripted-model tests for the loop | Create |
| `src/llm/backend/test_tool_hints.py` | tests for `matched_tools`/`matched_calls` | Modify |
| `src/llm/backend/test_web_app.py` | test the `"thinking"` compare variant | Modify |

Each source file stays under the 200-line repo cap.

---

### Task 1: Coverage set in `tool_hints.py`

Factor the existing matcher out of `routing_hints` so the same matches feed both the hint string and a deterministic coverage set. No behavior change to `routing_hints`.

**Files:**
- Modify: `src/llm/backend/tool_hints.py`
- Test: `src/llm/backend/test_tool_hints.py`

- [ ] **Step 1: Write the failing tests**

Add to `src/llm/backend/test_tool_hints.py`:

```python
from backend.tool_hints import matched_tools, matched_calls


def test_matched_tools_detects_compound_intent():
    t = matched_tools("give me the best move and the evaluation")
    assert t == {"best_move", "eval"}


def test_matched_calls_returns_canonical_calls():
    calls = matched_calls("play b3 and tell me the eval")
    assert calls["move"] == "<tool>move san=b3</tool>"
    assert calls["eval"].startswith("<tool>eval depth=")


def test_matched_tools_empty_on_no_intent():
    assert matched_tools("hi there") == set()
    assert matched_calls("") == {}
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest src/llm/backend/test_tool_hints.py -k matched -v`
Expected: FAIL — `ImportError: cannot import name 'matched_tools'`.

- [ ] **Step 3: Refactor `routing_hints` and add the coverage functions**

In `src/llm/backend/tool_hints.py`, replace the body of `routing_hints` with a call to a new `_match` helper, and add `matched_calls`/`matched_tools`. Replace this block:

```python
    msg = user_message or ""
    hits: list[tuple[str, str, str]] = []
    mv = _move_hint(msg)
    if mv:
        hits.append(mv)
    for tool, phrase, call, pat in _TRIGGERS:
        if tool == "best_move" and mv:
            continue  # naming a specific move overrides "what should I play"
        if pat.search(msg):
            hits.append((tool, phrase, call))
    if not hits:
        return ""
    lines = [f"- to {phrase}, call `{tool}`: {call}" for tool, phrase, call in hits]
    return ("\n\nROUTING HINT (the user's words map to these tools — call the tool, "
            "do not just describe it; ground your reply in the result):\n" + "\n".join(lines))
```

with:

```python
    hits = _match(user_message or "")
    if not hits:
        return ""
    lines = [f"- to {phrase}, call `{tool}`: {call}" for tool, phrase, call in hits]
    return ("\n\nROUTING HINT (the user's words map to these tools — call the tool, "
            "do not just describe it; ground your reply in the result):\n" + "\n".join(lines))


def _match(msg: str) -> list[tuple[str, str, str]]:
    """The intent matches as (tool, human phrase, canonical call). Shared by
    routing_hints (formats them) and matched_calls (coverage set)."""
    hits: list[tuple[str, str, str]] = []
    mv = _move_hint(msg)
    if mv:
        hits.append(mv)
    for tool, phrase, call, pat in _TRIGGERS:
        if tool == "best_move" and mv:
            continue  # naming a specific move overrides "what should I play"
        if pat.search(msg):
            hits.append((tool, phrase, call))
    return hits


def matched_calls(user_message: str) -> dict[str, str]:
    """tool name -> the canonical `<tool>…</tool>` call the user's words map to.
    The deterministic coverage set: every detected intent must be gathered before
    the staged loop narrates. Pure intent match (game-over handling lives in the
    caller)."""
    return {tool: call for tool, _phrase, call in _match(user_message or "")}


def matched_tools(user_message: str) -> set[str]:
    return set(matched_calls(user_message))
```

- [ ] **Step 4: Run to verify pass (and no regression)**

Run: `python -m pytest src/llm/backend/test_tool_hints.py -v`
Expected: PASS (all, including the pre-existing routing-hint tests).

- [ ] **Step 5: Commit**

```bash
git add src/llm/backend/tool_hints.py src/llm/backend/test_tool_hints.py
git commit -m "feat(thinking): expose matched_tools/matched_calls coverage set"
```

---

### Task 2: `thinking/prompts.py` — stage prompts + scoped payloads

Pure functions: the two stage system prompts and the deterministic context builders. No model calls.

**Files:**
- Create: `src/llm/backend/thinking/__init__.py`
- Create: `src/llm/backend/thinking/prompts.py`
- Test: `src/llm/backend/thinking/test_staged_loop.py`

- [ ] **Step 1: Create the package marker**

Create `src/llm/backend/thinking/__init__.py`:

```python
"""Serve-time staged thinking harness (Controller + Narrator)."""
```

- [ ] **Step 2: Write the failing tests**

Create `src/llm/backend/thinking/test_staged_loop.py`:

```python
import chess

from backend.game import Game
from backend.thinking.prompts import board_facts, facts_summary, build_controller_system, build_narrator_system


def test_board_facts_reads_live_board():
    g = Game()
    bf = board_facts(g)
    assert "turn=white" in bf and "legal_moves=20" in bf and "last_move=none" in bf


def test_facts_summary_compacts_results():
    assert facts_summary([]) == "(none yet)"
    assert facts_summary([("eval", "score: +0.30")]) == "eval→score: +0.30"


def test_controller_system_has_manifest_and_outstanding():
    s = build_controller_system("", None, "best move and eval", "", ["eval"])
    assert "AVAILABLE TOOLS" in s          # full manifest present (it can route)
    assert "DONE" in s and "OUTSTANDING" in s and "eval" in s


def test_narrator_system_has_no_tool_manifest():
    s = build_narrator_system("")
    assert "AVAILABLE TOOLS" not in s      # narrator cannot route
    assert "grounded" in s.lower()
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest src/llm/backend/thinking/test_staged_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: backend.thinking.prompts`.

- [ ] **Step 4: Implement `prompts.py`**

Create `src/llm/backend/thinking/prompts.py`:

```python
"""Dedicated stage prompts + scoped-context builders for the staged loop.

Each stage sees only what it needs (the focus principle): the Controller gets the
full tool/skill manifest + hints + outstanding coverage; the Narrator gets only the
gathered facts (it cannot route)."""
from __future__ import annotations

import chess

from ..inference import build_system_prompt, serving_skills_index
from ..tool_hints import routing_hints, skill_hints

CONTROLLER_HEADER = (
    "\n\nSTAGE — CONTROLLER. FIRST decide: is EVERY part of the user's goal satisfied "
    "by the facts gathered so far? If yes, output EXACTLY `DONE`. If not, output the "
    "single next `<tool>NAME arg=value</tool>` that gets a missing fact or performs the "
    "action. Output ONLY `DONE` or one tool call — never narrate."
)

NARRATOR_SYSTEM = (
    "You are the chess-coach narrator. Using ONLY the facts provided, write a short "
    "grounded reply to the user. Never invent numbers (a positive score favours White, "
    "negative Black). If there are no facts, answer directly or decline if the request "
    "is off-topic. End a coaching answer with one brief guiding question. Output no "
    "tool tags."
)


def board_facts(game) -> str:
    """Cheap deterministic situation read so the Controller need not spend a
    board_state step."""
    b = game.board
    turn = "white" if b.turn == chess.WHITE else "black"
    last = game.san_stack[-1] if game.san_stack else "none"
    check = "yes" if b.is_check() else "no"
    return f"turn={turn}, legal_moves={b.legal_moves.count()}, last_move={last}, check={check}"


def facts_summary(facts: list[tuple[str, str]]) -> str:
    """Compact tool→result list (the only memory the stages carry of this turn)."""
    if not facts:
        return "(none yet)"
    return "; ".join(f"{name}→{result.strip()}" for name, result in facts)


def build_controller_system(agent_overlay: str, plugin_context, user_message: str,
                            game_over: str, outstanding: list[str]) -> str:
    base = build_system_prompt(agent_overlay, plugin_context)
    hints = routing_hints(user_message, game_over) + skill_hints(user_message, serving_skills_index())
    out = ("\n\nOUTSTANDING (still required before DONE): " + ", ".join(outstanding)) if outstanding else ""
    return base + CONTROLLER_HEADER + hints + out


def build_narrator_system(agent_overlay: str) -> str:
    text = NARRATOR_SYSTEM
    extra = (agent_overlay or "").strip()
    if extra:
        text += "\n\nCUSTOMIZATION (tone only; never invent facts): " + extra
    return text


def controller_user(goal: str, facts: list[tuple[str, str]], board: str, outstanding: list[str]) -> str:
    lines = [f"User goal: {goal}", f"Board: {board}", f"Facts gathered: {facts_summary(facts)}"]
    if outstanding:
        lines.append("Still required: " + ", ".join(outstanding))
    lines.append("Next action (one <tool> call, or DONE):")
    return "\n".join(lines)


def narrator_user(goal: str, facts: list[tuple[str, str]]) -> str:
    return f"User goal: {goal}\nFacts:\n{facts_summary(facts)}\n\nWrite the grounded reply."
```

- [ ] **Step 5: Run to verify pass**

Run: `python -m pytest src/llm/backend/thinking/test_staged_loop.py -v`
Expected: PASS (the 4 prompt tests).

- [ ] **Step 6: Commit**

```bash
git add src/llm/backend/thinking/__init__.py src/llm/backend/thinking/prompts.py src/llm/backend/thinking/test_staged_loop.py
git commit -m "feat(thinking): stage prompts + scoped-context builders"
```

---

### Task 3: `thinking/parse.py` — parse the Controller output

**Files:**
- Create: `src/llm/backend/thinking/parse.py`
- Test: `src/llm/backend/thinking/test_staged_loop.py`

- [ ] **Step 1: Write the failing tests**

Append to `src/llm/backend/thinking/test_staged_loop.py`:

```python
from backend.thinking.parse import parse_controller


def test_parse_controller_tool_done_and_recovery():
    kind, call = parse_controller("<tool>eval depth=18</tool>")
    assert kind == "tool" and "eval" in call
    assert parse_controller("DONE") == ("done", None)
    assert parse_controller("done.") == ("done", None)
    # Gemma's native wrapper is recovered into a tool action
    k, c = parse_controller("Let me check. <tool_code>eval depth=18</tool_code>")
    assert k == "tool" and "<tool>eval" in c
    # prose that is neither a call nor DONE -> fail toward narrating (done)
    assert parse_controller("I think we are good here") == ("done", None)
    assert parse_controller("") == ("done", None)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest src/llm/backend/thinking/test_staged_loop.py -k parse_controller -v`
Expected: FAIL — `ModuleNotFoundError: backend.thinking.parse`.

- [ ] **Step 3: Implement `parse.py`**

Create `src/llm/backend/thinking/parse.py`:

```python
"""Parse a Controller turn into an action: a tool call or DONE.

Reuses inference.extract_call so the Controller benefits from the same recovery as
the single loop (<tool_code> normalize, malformed wrapper, hint-echo, stop-trim)."""
from __future__ import annotations

from ..inference import extract_call


def parse_controller(raw: str) -> tuple[str, str | None]:
    s = (raw or "").strip()
    if not s:
        return ("done", None)
    call = extract_call(s)                 # canonical <tool>…</tool> (recovered) or None
    if call is not None and "<tool>" in call:
        return ("tool", call)
    if s.upper().startswith("DONE"):
        return ("done", None)
    return ("done", None)                  # neither -> fail toward narrating
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest src/llm/backend/thinking/test_staged_loop.py -k parse_controller -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm/backend/thinking/parse.py src/llm/backend/thinking/test_staged_loop.py
git commit -m "feat(thinking): parse Controller output (tool|done) with recovery"
```

---

### Task 4: `thinking/loop.py` — `StagedLoop` orchestrator

The core. One Controller call per step, coverage-forced multi-tool, caps, then one Narrator call. Returns the same dict shape as `CoachLoop.respond` plus `trace`.

**Files:**
- Create: `src/llm/backend/thinking/loop.py`
- Test: `src/llm/backend/thinking/test_staged_loop.py`

- [ ] **Step 1: Write the failing tests**

Append to `src/llm/backend/thinking/test_staged_loop.py`:

```python
from backend.tools import ToolExecutor
from backend.thinking.loop import StagedLoop, MAX_STEPS


class ScriptedModel:
    """Returns scripted stage outputs in order; final extra output is the Narrator."""
    def __init__(self, steps):
        self.steps = list(steps)
        self.i = 0

    def generate(self, messages, max_new_tokens, stop):
        out = self.steps[min(self.i, len(self.steps) - 1)]
        self.i += 1
        return out


def _loop(steps, game=None):
    return StagedLoop(ScriptedModel(steps), ToolExecutor(game or Game(), None))


def _names(out):
    from backend.toolfmt import parse_call
    return [parse_call(c)[0] for c in out["tool_calls"]]


def test_one_tool_then_done():
    out = _loop(["<tool>eval depth=18", "DONE", "Equal here. Want the plan?"]).run([], "how am I doing?")
    assert _names(out) == ["eval"]
    assert out["reply"].endswith("?") and "<tool>" not in out["reply"]
    assert out["trace"] and out["trace"][-1]["stage"] == "narrator"


def test_multi_tool_model_driven():
    # empty required (no recognized intent) — the model chains tools itself
    out = _loop(["<tool>eval depth=18", "<tool>best_move depth=18", "DONE", "Here you go."]).run([], "tell me everything")
    assert _names(out) == ["eval", "best_move"]


def test_guaranteed_coverage_forces_missing_tool():
    # "best move and the evaluation" -> required {best_move, eval}; model DONEs early
    out = _loop(["<tool>best_move depth=18", "DONE", "DONE", "Summary."]).run([], "best move and the evaluation")
    assert set(_names(out)) == {"best_move", "eval"}   # eval force-routed despite premature DONE


def test_immediate_done_no_tools():
    out = _loop(["DONE", "Hello! Ask me about the position."]).run([], "hi there")
    assert out["tool_calls"] == [] and out["reply"]


def test_malformed_controller_recovers_then_done():
    out = _loop(["<tool_code>eval depth=18</tool_code>", "DONE", "Equal."]).run([], "evaluate it")
    assert _names(out) == ["eval"]


def test_dedup_with_nothing_outstanding_stops():
    out = _loop(["<tool>eval depth=18", "<tool>eval depth=18", "Done analysing."]).run([], "tell me everything")
    assert _names(out) == ["eval"]                     # repeat broke the loop


def test_cap_stops_at_max_steps():
    distinct = ["<tool>eval depth=18", "<tool>best_move depth=18", "<tool>threats depth=12",
                "<tool>review_move depth=15", "<tool>legal_moves", "<tool>list_pieces color=white",
                "<tool>board_state fields=all", "<tool>ask_chessbot query=hi",
                "<tool>load_fen fen=8/8/8/8/8/8/8/8 w - - 0 1", "<tool>undo"]
    out = _loop(distinct + ["reply"]).run([], "tell me everything")
    assert len(out["tool_calls"]) == MAX_STEPS


def test_game_over_no_analysis():
    g = Game()
    for san in ["f3", "e5", "g4", "Qh4#"]:
        g.move(san)
    out = _loop(["DONE", "That's checkmate — Black wins. New game?"], game=g).run([], "how am I doing?")
    assert out["tool_calls"] == [] and "checkmate" in out["reply"].lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest src/llm/backend/thinking/test_staged_loop.py -k "tool or done or coverage or cap or game_over or dedup" -v`
Expected: FAIL — `ModuleNotFoundError: backend.thinking.loop`.

- [ ] **Step 3: Implement `loop.py`**

Create `src/llm/backend/thinking/loop.py`:

```python
"""StagedLoop: the serve-time thinking harness. One Controller model call per step
(verify goal -> emit one tool or DONE); a deterministic coverage set force-routes any
required-but-missing tool so compound requests are guaranteed; then one Narrator call
writes the grounded reply. Same return shape as CoachLoop.respond, plus `trace`."""
from __future__ import annotations

from ..inference import (_build_window, contains_tool_call, _fallback_reply)
from ..tool_hints import matched_calls
from ..toolfmt import parse_call
from .parse import parse_controller
from .prompts import (board_facts, build_controller_system, build_narrator_system,
                      controller_user, narrator_user)

MAX_STEPS = 10


def _name(call: str) -> str:
    return parse_call(call)[0] or ""


class StagedLoop:
    def __init__(self, model, executor, agent_overlay: str = "", plugin_context=None, window=None) -> None:
        self.model = model
        self.executor = executor
        self.agent_overlay = agent_overlay
        self.plugin_context = plugin_context
        self.window = window or _build_window(model)

    def run(self, history: list[dict], user_message: str) -> dict:
        goal = user_message
        game_over = self.executor.game.over_status()
        coverage = {} if game_over else matched_calls(user_message)   # tool -> canonical call
        required = list(coverage)
        facts: list[tuple[str, str]] = []
        seen: set[str] = set()
        tool_calls: list[str] = []
        tool_results: list[str] = []
        trace: list[dict] = []

        for _ in range(MAX_STEPS):
            outstanding = [t for t in required if t not in seen]
            system = build_controller_system(self.agent_overlay, self.plugin_context,
                                              user_message, game_over, outstanding)
            convo = [{"role": "system", "content": system},
                     {"role": "user", "content": controller_user(goal, facts, board_facts(self.executor.game), outstanding)}]
            raw = self.model.generate(convo, max_new_tokens=96, stop=["</tool>", "</tool_code>"]).strip()
            trace.append({"stage": "controller", "output": raw[:120]})
            kind, call = parse_controller(raw)
            if kind == "done" or (call is not None and _name(call) in seen):
                if not outstanding:
                    break                              # goal covered (or nothing new) -> narrate
                call = coverage[outstanding[0]]         # backstop: gather a required-but-missing fact
            name = _name(call)
            result = self.executor.execute(call)
            facts.append((name, result))
            seen.add(name)
            tool_calls.append(call)
            tool_results.append(result)
            trace.append({"stage": "execute", "tool": name, "result": result[:120]})

        system_n = build_narrator_system(self.agent_overlay)
        convo_n = [{"role": "system", "content": system_n},
                   {"role": "user", "content": narrator_user(goal, facts)}]
        reply = self.model.generate(convo_n, max_new_tokens=160, stop=[]).strip()
        if contains_tool_call(reply) or not reply:
            reply = _fallback_reply(tool_calls, tool_results)
        trace.append({"stage": "narrator", "output": reply[:120]})

        _kept, ctx = self.window.fit(system_n, history, user_message)
        return {
            "reply": reply,
            "tool_call": tool_calls[-1] if tool_calls else None,
            "tool_result": tool_results[-1] if tool_results else None,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "turns": [{"role": "user", "content": user_message},
                      {"role": "assistant", "content": reply}],
            "context": ctx.as_payload(),
            "trace": trace,
        }
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest src/llm/backend/thinking/test_staged_loop.py -v`
Expected: PASS (all loop + prompt + parse tests).

- [ ] **Step 5: Commit**

```bash
git add src/llm/backend/thinking/loop.py src/llm/backend/thinking/test_staged_loop.py
git commit -m "feat(thinking): StagedLoop — coverage-forced staged tool loop"
```

---

### Task 5: Wire `CoachLoop.respond` to the engine toggle

`respond` gains a `thinking` parameter; when resolved to `staged` it delegates to `StagedLoop`. Default resolves from `CHESS_THINKING` env (default `single`). Lazy import avoids a circular import (`thinking` imports `inference`).

**Files:**
- Modify: `src/llm/backend/inference.py`
- Test: `src/llm/backend/thinking/test_staged_loop.py`

- [ ] **Step 1: Write the failing test**

Append to `src/llm/backend/thinking/test_staged_loop.py`:

```python
import os
from backend.inference import CoachLoop


def test_coachloop_delegates_to_staged_when_flag_set():
    loop = CoachLoop(ScriptedModel(["<tool>eval depth=18", "DONE", "Equal here."]),
                     ToolExecutor(Game(), None))
    out = loop.respond([], "how am I doing?", thinking="staged")
    assert "trace" in out and out["reply"]
    assert [parse_call(c)[0] for c in out["tool_calls"]] == ["eval"]


def test_coachloop_single_is_default(monkeypatch):
    monkeypatch.delenv("CHESS_THINKING", raising=False)
    # single loop: one decision (tool, stop-trimmed) then a plain reply
    loop = CoachLoop(ScriptedModel(["I'll check.\n<tool>eval depth=18", "Equal here."]),
                     ToolExecutor(Game(), None))
    out = loop.respond([], "how am I doing?")
    assert "trace" not in out                 # single path, unchanged shape
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest src/llm/backend/thinking/test_staged_loop.py -k coachloop -v`
Expected: FAIL — `respond() got an unexpected keyword argument 'thinking'`.

- [ ] **Step 3: Add the toggle to `respond`**

In `src/llm/backend/inference.py`, add `import os` at the top of the imports (after `from __future__ import annotations`):

```python
import os
```

Change the `respond` signature and prepend the delegation. Replace:

```python
    def respond(self, history: list[dict], user_message: str) -> dict:
```

with:

```python
    def respond(self, history: list[dict], user_message: str, thinking: str | None = None) -> dict:
        engine = thinking or os.environ.get("CHESS_THINKING", "single")
        if engine == "staged":
            from .thinking.loop import StagedLoop  # lazy: thinking imports inference
            return StagedLoop(self.model, self.executor, self.agent_overlay,
                              self.plugin_context, self.window).run(history, user_message)
```

(The existing single-loop body stays exactly as-is, now running only when `engine != "staged"`.)

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest src/llm/backend/thinking/test_staged_loop.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full backend suite (no regression)**

Run: `python -m pytest src/llm/backend -q`
Expected: PASS (all prior tests still green).

- [ ] **Step 6: Commit**

```bash
git add src/llm/backend/inference.py src/llm/backend/thinking/test_staged_loop.py
git commit -m "feat(thinking): CoachLoop.respond delegates to StagedLoop on thinking flag"
```

---

### Task 6: `web_app.py` — `thinking` flag + compare variant

Thread `thinking` through `chat`/`_run`; add a `"thinking"` variant that runs the same prompt through staged + single, the single run isolated on the mirrored board (reusing `base_executor` + `_mirror_base`) so it can't double-mutate the displayed game.

**Files:**
- Modify: `src/llm/backend/web_app.py`
- Test: `src/llm/backend/test_web_app.py`

- [ ] **Step 1: Write the failing test**

Append to `src/llm/backend/test_web_app.py`:

```python
def test_thinking_compare_returns_both_and_isolates_board():
    app = App(adapter=None)
    # staged engine drives the SFT loop; single engine runs on the mirror loop
    app.loop = CoachLoop(_StagedScript(["<tool>eval depth=18", "DONE", "Equal (staged)."]), app.executor)
    app.loop_mirror = CoachLoop(_SingleScript(["Equal (single).", ""]), app.base_executor)
    out = app.chat("how am I doing?", variant="thinking", thinking=None)
    assert set(out) == {"staged", "single", "state"}
    assert "staged" in out["staged"]["reply"] and "single" in out["single"]["reply"]
    assert app.game.board.fen() == chess.STARTING_FEN     # neither run advanced the real board
```

Add these scripted helpers near the top of `test_web_app.py` (below the existing imports):

```python
class _StagedScript:
    def __init__(self, steps):
        self.steps = list(steps); self.i = 0
    def generate(self, messages, max_new_tokens, stop):
        out = self.steps[min(self.i, len(self.steps) - 1)]; self.i += 1; return out


class _SingleScript(_StagedScript):
    pass
```

(The staged loop calls `generate` once per Controller step + once for the Narrator; the single loop calls it once for the decision + once for the reply. Both scripts just play their list in order.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest src/llm/backend/test_web_app.py -k thinking_compare -v`
Expected: FAIL — `chat() got an unexpected keyword argument 'thinking'`.

- [ ] **Step 3: Add `loop_mirror`, thread `thinking`, add the compare path**

In `src/llm/backend/web_app.py`:

(a) In `load_model`, after the two adapter loops are built, add a third (adapter ON, on the isolated board). Replace:

```python
                self.loop = CoachLoop(AdapterView(model, True), self.executor, ov, pc)
                self.loop_base = CoachLoop(AdapterView(model, False), self.base_executor, ov, pc)
```

with:

```python
                self.loop = CoachLoop(AdapterView(model, True), self.executor, ov, pc)
                self.loop_base = CoachLoop(AdapterView(model, False), self.base_executor, ov, pc)
                # adapter ON, on the isolated board — for the staged-vs-single compare
                self.loop_mirror = CoachLoop(AdapterView(model, True), self.base_executor, ov, pc)
```

(b) In `__init__`, add the mirror loop handle and the compare history. Replace:

```python
        self.loop_base: CoachLoop | None = None   # untrained base (adapter off)
```

with:

```python
        self.loop_base: CoachLoop | None = None   # untrained base (adapter off)
        self.loop_mirror: CoachLoop | None = None  # adapter on, isolated board (compare)
        self.history_single: list[dict] = []      # history for the single engine in compare
```

(c) In `reset`, clear the compare history. Replace:

```python
        self.history = []
        self.history_base = []
        return self.state()
```

with:

```python
        self.history = []
        self.history_base = []
        self.history_single = []
        return self.state()
```

(d) Thread `thinking` through `_run`. Replace:

```python
    def _run(self, loop: CoachLoop, history: list[dict], message: str) -> dict:
        result = loop.respond(history, message)
```

with:

```python
    def _run(self, loop: CoachLoop, history: list[dict], message: str, thinking: str | None = None) -> dict:
        result = loop.respond(history, message, thinking)
```

and extend the returned dict to carry the trace. Replace:

```python
        return {"reply": result["reply"], "tool_calls": result.get("tool_calls", []),
                "tool_results": result.get("tool_results", []),
                "context": result.get("context")}
```

with:

```python
        return {"reply": result["reply"], "tool_calls": result.get("tool_calls", []),
                "tool_results": result.get("tool_results", []),
                "context": result.get("context"), "trace": result.get("trace")}
```

(e) Add the `thinking` arg + the compare branch to `chat`. Replace:

```python
    def chat(self, message: str, variant: str = "sft") -> dict:
        if self.loop is None:
            return {"reply": f"(model not loaded: {self.model_error or 'no adapter'})",
                    "tool_calls": [], "tool_results": [], "state": self.state()}
        if variant == "both" and self.loop_base is not None:
            sft = self._run(self.loop, self.history, message)
            board = self.state()  # the visible board follows OUR model, snapshot before base runs
            self._mirror_base()   # base runs on a private copy — never touches the real board
            base = self._run(self.loop_base, self.history_base, message)
            return {"sft": sft, "base": base, "state": board}
        out = self._run(self.loop, self.history, message)
        return {**out, "tool_call": out["tool_calls"][-1] if out["tool_calls"] else None,
                "tool_result": out["tool_results"][-1] if out["tool_results"] else None,
                "state": self.state()}
```

with:

```python
    def chat(self, message: str, variant: str = "sft", thinking: str | None = None) -> dict:
        if self.loop is None:
            return {"reply": f"(model not loaded: {self.model_error or 'no adapter'})",
                    "tool_calls": [], "tool_results": [], "state": self.state()}
        if variant == "both" and self.loop_base is not None:
            sft = self._run(self.loop, self.history, message, thinking)
            board = self.state()  # the visible board follows OUR model, snapshot before base runs
            self._mirror_base()   # base runs on a private copy — never touches the real board
            base = self._run(self.loop_base, self.history_base, message, thinking)
            return {"sft": sft, "base": base, "state": board}
        if variant == "thinking" and self.loop_mirror is not None:
            staged = self._run(self.loop, self.history, message, thinking="staged")
            board = self.state()           # staged drives the visible board
            self._mirror_base()            # single runs on a private copy
            single = self._run(self.loop_mirror, self.history_single, message, thinking="single")
            return {"staged": staged, "single": single, "state": board}
        out = self._run(self.loop, self.history, message, thinking)
        return {**out, "tool_call": out["tool_calls"][-1] if out["tool_calls"] else None,
                "tool_result": out["tool_results"][-1] if out["tool_results"] else None,
                "state": self.state()}
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest src/llm/backend/test_web_app.py -v`
Expected: PASS (the new compare test + the existing base-isolation test).

- [ ] **Step 5: Commit**

```bash
git add src/llm/backend/web_app.py src/llm/backend/test_web_app.py
git commit -m "feat(thinking): web_app thinking flag + staged-vs-single compare variant"
```

---

### Task 7: `server.py` — pass the `thinking` flag

**Files:**
- Modify: `src/llm/backend/server.py`

- [ ] **Step 1: Pass `thinking` from the request body**

In `src/llm/backend/server.py`, replace the `/api/chat` handler line:

```python
                return self._json({"ok": True, **APP.chat(msg, str(body.get("variant", "sft")))})
```

with:

```python
                thinking = str(body.get("thinking", "")).strip() or None
                return self._json({"ok": True, **APP.chat(msg, str(body.get("variant", "sft")), thinking)})
```

- [ ] **Step 2: Verify import + handler integrity**

Run: `python -c "import sys; sys.path.insert(0,'src/llm'); import backend.server; print('ok')"`
Expected: prints `ok` (module imports cleanly).

- [ ] **Step 3: Commit**

```bash
git add src/llm/backend/server.py
git commit -m "feat(thinking): /api/chat accepts a thinking flag"
```

---

### Task 8: Frontend — thinking-mode toggle + compare toggle

Add two toggles next to the existing Dual toggle, wire `send()` to pick the variant/flag, and render the staged-vs-single comparison in the two existing panels. The existing `addThinking` panel already renders `tool_calls`→`tool_results`, so the staged trace shows with no extra work.

**Files:**
- Modify: `src/llm/gemma_chat_site/static/index.html`

- [ ] **Step 1: Add the two toggle controls**

In `src/llm/gemma_chat_site/static/index.html`, replace the dual-toggle block:

```html
        <div class="chat-dual-toggle">
          <label>Dual Model Comparison</label>
          <button class="toggle-switch on" id="dualToggle" onclick="ChatUI.toggleDual()"></button>
        </div>
```

with:

```html
        <div class="chat-dual-toggle">
          <label>Dual Model Comparison</label>
          <button class="toggle-switch on" id="dualToggle" onclick="ChatUI.toggleDual()"></button>
          <label style="margin-left:16px;">Thinking mode</label>
          <button class="toggle-switch" id="thinkToggle" onclick="ChatUI.toggleThinking()"></button>
          <label style="margin-left:16px;">Compare staged vs single</label>
          <button class="toggle-switch" id="compareToggle" onclick="ChatUI.toggleCompare()"></button>
        </div>
```

- [ ] **Step 2: Add the state vars + toggle handlers**

In the `ChatUI` IIFE, replace:

```javascript
  let dualMode = true;
```

with:

```javascript
  let dualMode = true;
  let thinkingMode = false;   // staged engine on/off
  let compareMode = false;    // staged-vs-single side by side
```

and replace `toggleDual` + the `return`:

```javascript
  function toggleDual() {
    dualMode = !dualMode;
    const toggle = document.getElementById('dualToggle');
    const panelBase = document.getElementById('panel-base');
    toggle.classList.toggle('on', dualMode);
    panelBase.style.display = dualMode ? 'flex' : 'none';
  }

  return { send, toggleDual };
```

with:

```javascript
  function toggleDual() {
    dualMode = !dualMode;
    if (dualMode) { compareMode = false; document.getElementById('compareToggle').classList.remove('on'); }
    document.getElementById('dualToggle').classList.toggle('on', dualMode);
    syncPanels();
  }

  function toggleThinking() {
    thinkingMode = !thinkingMode;
    document.getElementById('thinkToggle').classList.toggle('on', thinkingMode);
  }

  function toggleCompare() {
    compareMode = !compareMode;
    if (compareMode) { dualMode = false; document.getElementById('dualToggle').classList.remove('on'); }
    document.getElementById('compareToggle').classList.toggle('on', compareMode);
    syncPanels();
  }

  function syncPanels() {
    // The right panel is shown for either comparison; relabel its header to match.
    const showRight = dualMode || compareMode;
    document.getElementById('panel-base').style.display = showRight ? 'flex' : 'none';
    const ftHdr = document.querySelector('#panel-ft .chat-panel-header');
    const baseHdr = document.querySelector('#panel-base .chat-panel-header');
    if (compareMode) {
      ftHdr.innerHTML = '<span class="dot ft"></span> Staged thinking';
      baseHdr.innerHTML = '<span class="dot base"></span> Single-prompt';
    } else {
      ftHdr.innerHTML = '<span class="dot ft"></span> Gemma 4 with Trained Weights';
      baseHdr.innerHTML = '<span class="dot base"></span> Gemma 4 Base';
    }
  }

  return { send, toggleDual, toggleThinking, toggleCompare };
```

- [ ] **Step 3: Wire `send()` to choose variant + thinking flag**

In `send()`, replace:

```javascript
      const res = await api('/api/chat', { message: text, variant: dualMode ? 'both' : 'sft' });
      if (dualMode && res.sft) {
        renderReply(document.getElementById('msgs-ft'), res.sft, waitFT, 'ft-msg');
        renderReply(document.getElementById('msgs-base'), res.base, waitBase, '');
      } else {
        renderReply(document.getElementById('msgs-ft'), res, waitFT, 'ft-msg');
      }
```

with:

```javascript
      const variant = compareMode ? 'thinking' : (dualMode ? 'both' : 'sft');
      const thinking = thinkingMode ? 'staged' : 'single';
      const res = await api('/api/chat', { message: text, variant, thinking });
      if (compareMode && res.staged) {
        renderReply(document.getElementById('msgs-ft'), res.staged, waitFT, 'ft-msg');
        renderReply(document.getElementById('msgs-base'), res.single, waitBase, '');
      } else if (dualMode && res.sft) {
        renderReply(document.getElementById('msgs-ft'), res.sft, waitFT, 'ft-msg');
        renderReply(document.getElementById('msgs-base'), res.base, waitBase, '');
      } else {
        renderReply(document.getElementById('msgs-ft'), res, waitFT, 'ft-msg');
      }
```

Also update the loading skeleton so compare mode shows the right-panel skeleton. Replace:

```javascript
    if (dualMode) appendMsg(document.getElementById('msgs-base'), text, 'user');
    const waitFT = appendLoading(document.getElementById('msgs-ft'));
    const waitBase = dualMode ? appendLoading(document.getElementById('msgs-base')) : null;
```

with:

```javascript
    const showRight = dualMode || compareMode;
    if (showRight) appendMsg(document.getElementById('msgs-base'), text, 'user');
    const waitFT = appendLoading(document.getElementById('msgs-ft'));
    const waitBase = showRight ? appendLoading(document.getElementById('msgs-base')) : null;
```

- [ ] **Step 4: Verify the JS parses**

Run:
```bash
python - <<'PY'
import re, pathlib, subprocess, tempfile, os
html = pathlib.Path("src/llm/gemma_chat_site/static/index.html").read_text(encoding="utf-8")
b = re.findall(r"<script[^>]*>(.*?)</script>", html, re.DOTALL)[0]
f = tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8"); f.write(b); f.close()
r = subprocess.run(["node","--check",f.name], capture_output=True, text=True); os.unlink(f.name)
print("OK" if r.returncode==0 else r.stderr[:800])
PY
```
Expected: `OK`.

- [ ] **Step 5: Commit**

```bash
git add src/llm/gemma_chat_site/static/index.html
git commit -m "feat(thinking): web toggles for thinking mode + staged-vs-single compare"
```

---

### Task 9: Live smoke + flip-readiness

Confirm the staged engine grounds end-to-end on the real adapter and compare it to single. Manual (loads the model).

**Files:**
- Create (throwaway, outside repo): `A:/Download/thinking_smoke.py`

- [ ] **Step 1: Write the smoke script**

Create `A:/Download/thinking_smoke.py`:

```python
import os, sys
os.environ["CHESS_HF_ADAPTER"] = r"A:/Download/gemma4_chess_kaggle_adapter (1)"
sys.path.insert(0, r"A:/Download/llm_tool_calling_research_workspace/src/llm")
from backend.web_app import App  # noqa: E402

app = App(adapter=os.environ["CHESS_HF_ADAPTER"]); app.load_model()
if app.loop is None:
    print("MODEL NOT LOADED:", app.model_error); sys.exit(1)

def show(tag, out):
    calls = [c.split("\n")[-1][:50] for c in out.get("tool_calls", [])]
    leak = "<tool>" in out["reply"] or "<tool_code>" in out["reply"]
    print(f"\n[{tag}] calls={calls} leak={leak}\n  {out['reply'][:200]}", flush=True)

app.reset()
show("single", app.chat("give me the best move and the evaluation", variant="sft", thinking="single"))
app.reset()
show("staged", app.chat("give me the best move and the evaluation", variant="sft", thinking="staged"))
print("\n=== DONE ===", flush=True)
```

- [ ] **Step 2: Run it**

Run: `cd /a/Download && timeout 300 python thinking_smoke.py 2>&1 | tail -20`
Expected: both replies grounded, `leak=False`. The **staged** run shows `calls` containing BOTH `best_move` and `eval` (coverage guarantee); the single run may show only one (the bug we're fixing).

- [ ] **Step 3: Record the result**

Append a short Evidence note to `docs/2026-06-12-serve-audit-readiness.md` (or a new dated report) with the two outputs and whether staged covered both tools. Do NOT flip the `CHESS_THINKING` default yet — leave it `single`; the toggle proves the difference live first.

- [ ] **Step 4: Commit the evidence**

```bash
git add docs/2026-06-12-serve-audit-readiness.md
git commit -m "docs(thinking): live smoke — staged vs single coverage evidence"
```

---

## Notes for the implementer

- **Circular import:** `thinking/*` imports from `inference`/`tool_hints`/`toolfmt`. `inference` must import `StagedLoop` **lazily** (inside `respond`, as in Task 5) — never at module top. Don't add a top-level `from .thinking... import` to `inference.py`.
- **No engine needed for tests:** `ToolExecutor(Game(), None)` — analysis tools return `error: engine_unavailable`, which is a valid gathered "fact" for the loop's control flow. Tests assert routing/coverage, not engine numbers.
- **`force_tool` is just `coverage[name]`** — the canonical call string the hint layer already produces (`matched_calls`). No separate arg-building code.
- **Return shape parity:** `StagedLoop.run` returns the same keys as the single `respond` plus `trace`; `web_app`/`server`/frontend already tolerate extra keys.
- **Default stays `single`.** The flip to `staged` is a one-line env default change after live validation — not part of this plan.
