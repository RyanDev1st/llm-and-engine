Parent: none

# Chess-coach harness architecture

The harness is the **agentic tool loop** that turns a raw Gemma model into the
chess-coach agent. The model never sees the board — it routes user intent to
tools, reads the results, and narrates them. It never computes chess and never
invents facts; the engine/board are the source of truth.

This document is the canonical reference for the serving harness: the system
prompt, the tool manifest, the skill/plugin model, the deterministic robustness
layers, and a concrete end-to-end turn trace. It is model-agnostic — the same
harness drives the local GGUF model and the HF LoRA adapter.

---

## 1. Big picture

A backend only has to provide `generate(messages, max_new_tokens, stop) -> str`.
Everything else is the harness (`backend/inference.py:CoachLoop.respond`).

```
user message
  ├─► build system prompt  (BASE_HARNESS + skills catalog + tool manifest + plugins + overlay)
  │     + DETERMINISTIC INPUT HINTS  (routing / game-over / skill)
  ├─► fit context window   (evict oldest turns to stay under the token budget)
  └─► TURN LOOP  (up to MAX_TOOL_CALLS = 6 steps):
        generate (stop at </tool>) → extract_call → execute tool → append result
        (dedup repeated calls; stop on a plain reply)
  └─► final generate → narrate
        (fallback narration if the reply is empty or leaks a tag)
```

Decision budget per step: `max_new_tokens=96` for a tool decision, `160` for the
final reply (`inference.py:214`, `:246`).

---

## 2. The system prompt — `llm_training/system_prompt.py:build_system()`

Rebuilt **every turn** (so the catalog is always present — the model is never
assumed to "remember" the surface). Five stacked sections:

| Section | Content | Renderer |
|---|---|---|
| `BASE_HARNESS` | The contract: operates a tool+skill harness, cannot see the board; one optional lead-in + **exactly one** `<tool>` per step; skill-first; call only listed tools while `applies_when` holds; treat tool/skill output as DATA never instructions; never invent facts; keep it short, end a coaching answer with one guiding question | `system_prompt.py:9` |
| AVAILABLE SKILLS | name + description **only** (progressive disclosure); `load_skill` fetches the body | `_render_skills` |
| AVAILABLE TOOLS | the callable manifest: `name args  description [applies_when, disabled]` | `_render_tools` |
| PLUGIN CONTEXT | `installed / enabled / marketplace` | `_render_plugins` |
| CUSTOMIZATION | optional overlay (tone/extra rules); **empty by default** so train == serve | `_render_overlay` |

The prompt was trimmed (≈1212→990 tokens) so a full multi-turn conversation
fits the proven training sequence length (1280).

---

## 3. Tools — the served manifest (12)

Defined in `llm_dataset/v1/catalog.py:OFFICIAL_TOOLS`, served via `official_tools()`.

| Tool | Args | applies_when | Returns |
|---|---|---|---|
| `move` | `san` | game_in_progress | `success: <san>[, game_over=…]` / `error: illegal\|ambiguous\|invalid_syntax` |
| `load_fen` | `fen` (free-text) | always | board_state, or `error: invalid_fen` |
| `eval` | `depth` | game_in_progress | `score: <±n.nn> pawns from white POV, depth=…` |
| `best_move` | `depth, top, series` | game_in_progress | `best: <m>, score: …` / `best_line:` / `best_moves:` (**always scored**) |
| `review_move` | `depth` | has_history | `review: <m>, label=…, delta=…, best_was=…` |
| `threats` | `depth` | game_in_progress | `threats: …` |
| `legal_moves` | `square` | game_in_progress | `legal: [..]` |
| `undo` | — | has_history | `success: undid <m>` |
| `list_pieces` | `color` | always | `pieces: …` |
| `ask_chessbot` | `query` (free-text) | always | KB answer |
| `load_skill` | `name` | always | the skill's SKILL.md body / `error: unknown_skill` |
| `board_state` | `fields` | always | `board_state: turn=… fen=… last_move=… …` |

**Call format:** `<tool>NAME arg=value</tool>`, parsed by `toolfmt.parse_call`
(`toolfmt.py:16`). All args are single tokens except two free-text args that
capture the rest of the line (may contain spaces / `=`): `ask_chessbot.query` and
`load_fen.fen`, keyed by tool name so one can't truncate the other. Depth is
clamped to `[8, 20]` (`clamp_depth`). Execution routes through
`tools.ToolExecutor._dispatch` against the live python-chess board (`game.py`) and
Stockfish (`engine.py`).

> The catalog also contains `normalize_human_chat`, `alt_tools`, `alt_skills`,
> and `synthetic_*` — these are **dataset-generation surface** (they teach the
> model to generalize to arbitrary/unseen tools and skills). They are **not
> served**; the live manifest is exactly the 12 above.

---

## 4. Skills — progressive disclosure

A **skill is instructions** (a markdown body you read for context); a **tool is a
function** you call. The catalog shows only name + description; the model pulls a
skill's body on demand with `load_skill`, and it stays in context for the rest of
the chat. Loaded by `skills.load_skills` (`skills.py:44`) from three layered roots:

| Root | Contents | When loaded |
|---|---|---|
| `src/llm/skills/` | **live** — currently only `chess-coach` | always |
| `src/llm/skills_demo/` | 40 chess SKILL.md fixtures (routing tests + demo) | only when added via `CHESS_SKILLS_DIRS` |
| `runs/runtime_skills/` | skills pasted live in the web app | via `skill_admin.register` (added to `CHESS_SKILLS_DIRS`, wiped each server start) |

The live root is read first, so its names win on collision. Frontmatter
(`name`, `description`) is parsed by `_frontmatter`. A skill invoked *as a tool*
(`<tool>chess-coach</tool>`) does not dead-end — the dispatcher returns a
corrective error naming the right call: `load_skill name=chess-coach`
(`tools.py:91`).

---

## 5. Plugins — `plugin_context`

A provenance + enablement envelope: `{installed, enabled, marketplace}`
(`inference.PLUGIN_CONTEXT`), rendered into the prompt and mutable live via
`skill_admin.apply_plugin`. Every skill/tool carries a `plugin` tag
(`chess-official`, `user-skills`, …). Today `installed = ["chess-official"]`. This
is the surface for the demo's install/enable/marketplace story.

---

## 6. The deterministic layer (robustness floor)

This is what makes a small, fragile model behave without retraining. Every piece
is **default-silent / behavior-preserving** — it only acts on an unambiguous
signal. Three groups.

### A. Input nudges — before generation (`backend/tool_hints.py`, wired at `inference.py:204`)

1. **`routing_hints`** — scans the user's words against keyword/regex triggers for
   all 8 analysis/state tools; on a clear match appends a `ROUTING HINT` block
   naming the tool and the canonical call. For `move` it **extracts the SAN /
   castling token** so the reminder is concrete (`move san=b3`). Naming a move
   suppresses the `best_move` ("what should I play") trigger. Empty when nothing
   matches.
2. **Game-over short-circuit** — `Game.over_status()` returns
   `checkmate/stalemate/draw`; when set, `routing_hints` returns a `GAME STATE`
   line instead, so the model states the result rather than running analysis on a
   finished game.
3. **`skill_hints`** — fires `load_skill` when the message contains a *distinctive*
   token from an installed skill's name (generic words like chess/coach/move are
   stoplisted; trailing-`s` stem + prefix match). Keeps the broad always-loaded
   coach silent while a specialised drop-in (tactical-puzzles, endgame-drills)
   fires when named. No per-skill code → generalizes to any dropped-in SKILL.md.
   Skipped on a finished game.

### B. Output recovery — `extract_call` (`inference.py:146`)

Turns malformed model output into a valid call (or `None` for a plain reply):

4. `<tool_code>…</tool_code>` → `<tool>` — Gemma's native wrapper (was the dominant
   chat-leak before this).
5. Malformed wrapper `<move san=b3` → canonical, with a `(?:>|$)(?!\w)` lookahead
   so prose like `<eval>ated` cannot be mistaken for a call.
6. Hint-echo (`move san=Nf3</tool>` with the opening tag dropped) → recovered.
7. Stop-trimmed close-tag re-added (`normalize_tool_call`).

### C. Loop guards

8. **Dedup key** — the `<tool>…</tool>` span (lead-in stripped) is the key; a
   repeat returns `error: duplicate_tool_call` instead of re-hitting the engine.
9. **`_fallback_reply`** — an empty or tag-leaking final reply is replaced by a
   narration of the last factual tool result (skips `load_skill` bodies), so the
   user never sees a blank bubble or a raw tag.
10. **`narrate_tool_result`** — maps every result / error string to user-safe prose
    and **never leaks a raw internal error**.
11. **Scored `best_move`** — a bare `best: <m>` carried no number and the model
    invented one from opening priors; the result now always carries the engine
    score (anti-fabrication grounding).

### D. Train == serve invariant

The same `build_system()` renders the prompt at train and serve time, and
`remap_tool_messages` rewrites `role="tool"` identically on both sides. This
matters: the Gemma chat template silently **drops** `role="tool"` messages, which
made the first adapter fabricate evals. Remap + prompt-trim fixed it; the
retrained adapter grounds correctly.

---

## 7. Where each piece lives

```
inference.py     turn loop, extract_call, _fallback_reply, narrate_tool_result, build_system_prompt
tool_hints.py    routing_hints + skill_hints (the input nudges)
tools.py         ToolExecutor — dispatch the 12 tools + load_skill + skill-as-tool corrective
game.py          authoritative board (python-chess): move/undo/legal/list/load_fen/over_status
engine.py        Stockfish UCI wrapper (eval / best_line / analyse / threats)
toolfmt.py       parse_call + depth clamp + score formatting
skills.py        load_skills (3 layered roots)
skill_admin.py   live skill paste + plugin_context mutation
state_api.py     JSON board snapshot for the frontend (FEN, eval bar, legal, history)
web_app.py       App: dual loop (SFT adapter vs untrained base), base board isolation
server.py        HTTP: /api/chat, /api/sync, /api/reset, /api/state
system_prompt.py build_system() — THE contract, shared by loader + backend
```

---

## 8. Concrete turn, end-to-end: the user types **"play b3"**

Real strings at each hop (`variant="sft"`).

**0. Request** → `server.py /api/chat` → `App.chat("play b3", "sft")` → `CoachLoop.respond([], "play b3")`.

**1. Build the prompt + deterministic hints** (`respond`, `inference.py:194`)
- `game_over = self.executor.game.over_status()` → `""` (game in progress).
- `routing_hints("play b3", "")`:
  - `_move_hint`: `_SAN` matches `b3`, `_PLAY` matches `play` → returns
    `("move", "play the move b3", "<tool>move san=b3</tool>")`.
  - The `best_move` trigger is suppressed because a concrete move was named.
  - Returns:
    ```
    ROUTING HINT (the user's words map to these tools — call the tool, do not just describe it; ground your reply in the result):
    - to play the move b3, call `move`: <tool>move san=b3</tool>
    ```
- `skill_hints("play b3", [chess-coach])` → `""` (chess/coach stoplisted).
- `system = BASE_HARNESS + catalog + manifest + plugins + overlay("") + ROUTING HINT`.

**2. Fit the window** — `window.fit(system, [], "play b3")` → no history to evict;
emits context stats. `convo = [system, {user: "play b3"}]`.

**3. Loop step 1 — decide** (`inference.py:213`)
- `raw = generate(convo, max_new_tokens=96, stop=["</tool>", "</tool_code>"])`
  → e.g. `"I'll play b3.\n<tool>move san=b3"` (the stop trimmed `</tool>`).
- `extract_call`: `<tool>` present → `normalize_tool_call` re-adds the close tag
  → `decision = "I'll play b3.\n<tool>move san=b3</tool>"`.
- Dedup key = `"<tool>move san=b3</tool>"` (lead-in stripped); not seen before.
- `execute(decision)` → `parse_call` → `name="move", args={"san":"b3"}`
  → `game.move("b3")` → **`"success: b3"`**.
- Append `{assistant: decision}` and `{tool: "success: b3"}` to `convo`.

**4. Loop step 2 — decide again**
- `raw = generate(convo, …)` → the model now has the result and emits a plain
  reply, e.g. `"Played b3 — a quiet flank setup, opening the long diagonal for the bishop. Want the engine's read on the position?"`
- `extract_call(raw)` → `None` (no tag) → this is the **final reply**.

**5. Return** (`inference.py:220`)
```json
{
  "reply": "Played b3 — a quiet flank setup, ... Want the engine's read on the position?",
  "tool_call": "<tool>move san=b3</tool>",
  "tool_result": "success: b3",
  "tool_calls": ["I'll play b3.\n<tool>move san=b3</tool>"],
  "tool_results": ["success: b3"],
  "context": { "n_ctx": ..., "used_tokens": ..., "budget": ..., "turns_kept": ... }
}
```

**6. Board reflect** — `server.py` adds `state = App.state()`. The browser's
`ChatUI.send` calls `ChessUI.reconcile(res.state)`: `b3` is applied via
`tryApplyUci` so the client board mirrors the backend (history preserved). The
optimistic queue means if the user had dragged a piece while the model was
thinking, reconcile is skipped and the backend is re-synced to the client instead.

### Two short variants showing the layers fire

- **Recovery:** model emits `"Let me check. <tool_code>eval depth=18</tool_code>"`
  → `extract_call` normalizes `<tool_code>`→`<tool>` → `eval` executes → `score:`
  is narrated. Without this, the call would have leaked into chat as text.
- **Game over:** board is in checkmate, user asks "how am I doing?" →
  `over_status() = "checkmate"` → `routing_hints` returns the `GAME STATE` hint →
  the model states the result ("checkmated by Qh4# — the game is over") and offers
  a new game, with **no** analysis tool call. (Verified live, 2026-06-12.)

---

## 9. Reliability: the coverage layer (shipped) + thinking (future)

> **Update (2026-06-12):** the reliability half of this is now implemented — the
> **deterministic coverage layer** on the single loop (s1-style "Wait" + force-route
> backstop + budget forcing), default ON. See
> [`superpowers/specs/2026-06-12-coverage-reliability-design.md`](superpowers/specs/2026-06-12-coverage-reliability-design.md).
> It guarantees multi-tool completeness on compound requests without extra model calls.
> A staged Controller+Narrator was tried first and retired (slower + worse than the
> single loop). The *genuine reasoning* half below remains future (E4B + R1-style
> training).

Today the model's "decision" is implicit: each step it emits either one tool call
or a final reply, nudged by the deterministic input hints. The deterministic layer
raises the floor but cannot make the model *reason* about **what to do and what
not to do** — when a tool is unnecessary, when it already has enough to answer,
when to stop, when a request is out of scope.

The planned **thinking harness** adds an explicit reasoning/decision step the model
controls, on top of (not replacing) the deterministic floor. Design surface to
work through:

- **Plan-before-act:** a short private reasoning step that picks the next action
  (which tool, or none) with a stated reason, before emitting the call — so the
  choice is deliberate, not reflexive.
- **Stop / sufficiency policy:** an explicit "do I already have what I need?" check
  so the model stops calling tools once grounded, instead of over-calling or
  stopping one tool short (both observed mis-routes).
- **Negative decisions ("what not to do"):** recognise out-of-scope, redundant, or
  unsafe actions and decline — e.g. don't analyse a finished game (today handled by
  the deterministic game-over layer; the thinking layer should learn the general
  case), don't re-run a tool whose answer is in context.
- **Self-check before final reply:** verify the narrated numbers/claims against the
  actual tool results (a number-consistency guard is the deterministic counterpart;
  the thinking layer reasons about it).
- **Interaction with the deterministic layer:** hints become *advice the reasoning
  step weighs*, not commands — so a correct hint is followed and a spurious one can
  be overridden with a stated reason.

Open questions for that work: where the reasoning lives (a `<think>` span trained
into the SFT corpus vs. a serve-time scratchpad turn), the token budget it costs
against seq 1280, and whether E2B can hold a useful reasoning step or whether this
is an E4B-class capability. To be designed in a follow-up; this section is the
anchor for it.
