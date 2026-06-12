Parent: implementation.md

# Conversational agent shape — "coding agent, for a chess chat"

## Status

Design approved (2026-06-06). Enriches the harness so the agent *converses* like a coding agent: narrate intent, call a tool, narrate, answer with a guiding follow-up. Decisions: **persistence = backend auto-save** (not a model tool); **narration = one lead-in sentence before each tool call**.

## Layer split (do not conflate)

| Concern | Owner | Trained in SFT? |
| --- | --- | --- |
| Decide skill/tool, lead-in narration, final answer + guiding question | model (tokens) | YES |
| Stream tokens (lead-in shows before tool finishes) | backend serving loop (SSE) | no |
| Save conversation each turn (local == backend on one demo machine) | backend side-effect | no |
| "Thinking"/tool-running spinner, board wireframe | frontend | no |

The model emits tokens; the server streams + persists. The model does not save or stream.

## Target turn shape (what the SFT teaches)

```
user>      hey, how's my game looking?
assistant> Let me load my coaching skill and read the board.        # lead-in (streams)
           <tool>load_skill name=chess-coach</tool>
tool>      <chess-coach SKILL.md body: when-to-use, ordered steps, a constraint>
assistant> First, the current position.
           <tool>board_state fields=all</tool>
tool>      board_state: turn=white, fen=..., check=no, legal_count=31
assistant> Now the engine's read.
           <tool>eval depth=15</tool>
tool>      score: +1.50 pawns from white POV, depth=15
assistant> You're in good shape — White's about +1.5, a clear edge but not winning, mostly from
           more active pieces. Want me to map the plan to convert it, or check Black's threats first?
```

Three SFT changes vs the current corpus:
1. **Lead-in narration** — one short sentence before each tool call. Streams immediately (responsiveness). Represented as prose preceding the single `<tool>` in the same assistant turn.
2. **Guiding-question final** — brief grounded assessment (cp translated to plain words) + one follow-up question that suggests what to ask next. Encouraging, not persona fluff. Applies to coaching/analysis finals; greetings (J) and pure-knowledge (ask_chessbot/I) stay statement-only.
3. **Real skill bodies** — `load_skill` result is a realistic multi-line SKILL.md (name, when-to-use, ordered steps, a constraint), and the steps actually follow it; sometimes a non-chess-coach skill is loaded (ties to the dynamic-SKILL.md objective).

## Contract / code changes

1. `system_prompt.BASE_HARNESS`: action turn = "one short lead-in sentence, then exactly ONE `<tool>` call"; final = brief + may end with a guiding question; still no XML in the final.
2. `validate.py`: detect tool calls by **search** (prose-before-tool allowed); enforce **exactly one** `<tool>` per assistant turn; the final = last assistant turn with **zero** `<tool>`; keep legality + skill-order + manifest checks working on the extracted call.
3. `renderer/chess.py` + `renderer/universality.py`: emit lead-ins + guiding-question finals + real skill bodies; load varied skills.
4. backend: serving loop streams lead-in → runs tool on `</tool>` → continues (`parse_call` already search-based); **auto-save** each turn to disk; implement `load_skill` returning the body (Phase 3 parity).
5. Regenerate v1.2 + re-run QC gate (legality, leak, personas, declared-tools) PLUS new checks: every action turn has exactly one tool; coaching finals end with a question.

## Why

Current data trains *when* to call tools, not *how to converse*. This makes the agent feel like a coach (and like a coding agent): it shows its work, stays responsive via streaming lead-ins, and ends with a nudge so the user knows what to ask next — while staying grounded in real tool results.

## Final contract decisions (2026-06-06) — skills, diversity, flexibility

- **`load_skill` is a tool** (a helper-function); a **skill is text you read for context**. No separate `<skill>` tag — the distinction is conceptual: skills live in their own catalog section, tools in the manifest. (Done in BASE_HARNESS.)
- **Progressive disclosure (Anthropic Agent Skills):** the catalog of ALL available skills (name+description) is always in the system context and re-read every turn; the model picks by description; `load_skill` pulls a body which then persists.
- **One tool call per inference step** (per assistant message), MANY across the agentic loop — like a coding agent (act → read result → act). Refined 2026-06-07 from an earlier "multiple per turn" reading. `one_tool_per_message` rule in `validate.py`; keep `no_exact_duplicate` + `max_six_tool_calls`. (Done.)
- **Cross-domain skill diversity (THE open data fix):** measured diversity — index offers **2,737 distinct skills**, but the agent **only ever LOADS 2** (chess-coach ×49,638, hood-human-chat ×761). Severe bias. Must generate skill-routing rows across many domains (code, math, writing, cooking, novel synthetic SKILL.md), correct skill ~uniform, + reject rows (wrong/irrelevant skill loaded, or acted before load). Target: loaded-skill diversity 2 → hundreds. Chess stays the primary task domain; skill selection becomes general.

Done so far: `build_system` catalog rendering, loader serialization, validator (lead-in + multi-call + legality). **Remaining: renderer cross-domain skill routing + multi-skill loads + real skill bodies → regenerate → re-audit (loaded-skill diversity).**
