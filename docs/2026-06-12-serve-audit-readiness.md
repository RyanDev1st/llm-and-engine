Parent: none

# Chess-coach serve/UI audit — readiness for tomorrow's test & audit

## Status

Gap/bug hunt of the serve + web-UI stack (the chat/board work from commits `68114590`→`3d1e505d`). Method: 3 parallel static code reviewers (serve-flow, board-sync, frontend-JS) **plus a live end-to-end user-flow** against the running model + Stockfish. The live flow caught the dominant bug the static pass missed.

**Fixed + verified this session** (see Evidence): the `<tool_code>` wrapper mismatch (dominant chat failure), routing hints, move/FEN/board-reflect, malformed-call recovery, skill-as-tool corrective, scored `best_move`, raw-error leaks, plus a safe hardening sweep (dedup key, parse_call guard, api ok-check, renderReply guard, folder-drop fix, new-game backend reset).

**Closed after the audit** (commits `64529468`, `f241cb4a`, `8c9ae5e9`): #1 drag-during-chat (optimistic queue — moveSeq guard, board never freezes), #2 dual-mode base loop (own ToolExecutor/Game, mirrored per turn — base can't touch the real board), #3 extract_call `$`-anchor (lookahead), #5 partial-board sync (atomic `load_uci_moves` + client aborts on non-200). Plus two NEW deterministic layers: game-over short-circuit and a generic skill-router. **#4 (castling/ep + underpromotion) deferred — awaiting decision.** Live smoke (real adapter) after the changes: eval/move/game-over all grounded, 0 leaks. 39 backend tests pass.

**Open** (was 5; now 1): #4 only — listed in **Next**.

Honest read: the harness is now far more reliable than the raw E2B weights give (most routing slips self-heal). The remaining model-side limits (occasional mis-route, soft magnitude wording) are E4B-retrain items, not serve bugs.

## Scope

Files: `src/llm/backend/{inference,tools,toolfmt,game,server,web_app,state_api,tool_hints,model_hf}.py`, `src/llm/gemma_chat_site/static/index.html`, `src/llm/skills/chess-coach/SKILL.md`. Out of scope: training, the adapter weights.

## Evidence

Commands:
- Static review: 3 `code-review-expert` agents (serve-flow / board-sync / frontend-JS).
- Live flow: `python A:/Download/audit_flow.py` against `backend.server` on the loaded adapter (9 scenarios: multi-turn coaching, model move, load_fen, mate-in-1, game-over, skill drop-in, dual-mode, context growth, edge cases).
- Tests: `python -m pytest src/llm/backend src/llm/llm_dataset/v1/tests -q` → **133 passed**.

What the live flow confirmed WORKING:
- Move plays + board reflects (history grows); `load_fen` puzzle setup; mate-in-1 → "Re8#, forced mate in one" (grounded); dual-mode board follows SFT; context meter climbs correctly; off-topic declined; bad FEN → 400.
- After the `<tool_code>` fix: "why is that the best?"/"did I blunder?"/"any threats?" route to eval/review_move/threats with grounded replies and **no leak** (previously all leaked `<tool_code>…` as text).

Dominant bug found + fixed (`fa4748f4`): the model emits Gemma's native `<tool_code>…</tool_code>` wrapper for many calls; the harness only recognized `<tool>`, so those calls leaked into chat and never executed. Now normalized to `<tool>` (+ stop token + truncate + contains-check).

## Next (OPEN findings — triage in priority order)

1. **[DONE — `64529468`] CRITICAL — drag-during-chat desync.** `handleBoardClick`/`autoOpponent` aren't gated on the chat `busy` flag (it lives in ChatUI, board in ChessUI). User drags a piece while a chat is in flight → `/api/chat` runs on a stale synced board, reconcile then clobbers the drag. Repro: play e4, immediately drag e5, send a message. Fix: expose `ChatUI.isBusy()` and early-return in `handleBoardClick`; **UX decision needed** — freeze the board during a model turn, or queue the drag. (Why deferred: UX call + cross-module wiring; not a blind one-liner.)

2. **[DONE — `f241cb4a`] CRITICAL — dual-mode base loop shares `APP.game`.** In `web_app.chat("both")` the base loop runs `_run` on the same `executor.game` after the SFT snapshot; if base calls `move`/`undo`/`load_fen` it mutates the real board. Board display self-heals (next `/api/sync` re-mirrors the client), but `/api/state` polled between response and next sync (the eval bar) can show the base-corrupted position. Fix: give `loop_base` its own `ToolExecutor` over a separate `Game`. (Why deferred: real fix, needs a test that base never touches `APP.game`.)

3. **[DONE — `82aa2077`] HIGH — `extract_call` `$`-anchor false positive.** `_MALFORMED` matches `<toolname …$` to recover a stop-stripped call (needed for `<move san=b3`), but also fires on prose like `<eval>…` → an uninstructed tool runs. Tension: removing `$` breaks the real truncated-move recovery. Fix: require the malformed match to carry plausible args or be a known no-arg tool, OR only recover when the segment is at end-of-string AND preceded by a play/analysis lead-in. Needs careful unit cases. (inference.py `_MALFORMED`.)

4. **[OPEN — awaiting decision] HIGH — `reconcile` `sameBoard` ignores castling/en-passant + underpromotion.** `sameBoard` compares only FEN fields 0–1 (layout+turn); a `load_fen` that changes only castling/EP is seen as "no change" and the client keeps stale rights. And `tryApplyUci` auto-queens, so an underpromotion takes the `loadFEN` fallback and **loses move history** (`originFromStart` flips false permanently → undo/review disabled for the session). Fix: compare FEN fields 0–3; pass the promotion piece through `tryApplyUci`. (index.html `reconcile`/`tryApplyUci`.)

5. **[DONE — `f241cb4a`] HIGH — `/api/sync` partial-board on failure + client ignores the status.** `load_uci_moves` resets then replays; on an illegal move it returns `False` leaving a partial board, and `ChatUI.send` doesn't check the sync response before calling `/api/chat`. Fix: save/restore the board in `load_uci_moves` on failure; in `send`, abort the chat if sync `!ok`. (Partially mitigated now: `api()` throws on non-200, so a 400 sync now aborts `send` via the catch — verify this is the desired UX.)

MEDIUM (do after the above): `_review` leaves the board one ply back if the engine throws between `pop()`/`push()` (wrap in try/finally or analyse on a copy); `load_fen` routing-hint regex over-fires on FEN-lookalike chat; halfmove clock hardcoded `0` in `getFEN` (50-move rule never triggers); empty `/api/chat` message returns 500 not 400; verbose model preamble can exhaust the 96-token decision budget mid-call → `invalid_syntax`.

MODEL-side (E4B/retrain or SKILL.md — not serve bugs): occasional mis-route on terse phrasings without the hint; soft magnitude wording ("slightly better" for a clear edge); skill drop-in not always loaded (model answers directly instead of `load_skill`); game-over narration asks for the FEN instead of recognizing mate.

Recommended order tomorrow: run the suite (133 should pass) → fix #1 and #2 (the two CRITICALs, with the UX decision on #1) → #3–#5 → MEDIUM. The MODEL-side items wait for the E4B run.
