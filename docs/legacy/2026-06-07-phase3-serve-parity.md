Parent: implementation.md

# Phase 3 — backend harness parity (serve == train)

## Status

DONE (code + tests, 2026-06-07). The backend now serves the *exact* contract the
model was trained on: same `build_system()` renderer, real `load_skill`
execution, progressive disclosure (no skill-body pre-stuffing), one-tool-per-step
loop that handles lead-in narration, optional customization overlay (default
empty), GGUF-only. Verified by a no-GGUF/no-Stockfish end-to-end smoke. Not yet
run against a trained model (that is Phase 4).

## Scope

Tasks 9–14 of `implementation.md`. Built on the prior backend baseline
(`ef8739c9`: board_state, best_move MultiPV, CoachLoop).

## Evidence

- **Task 9 — overlay (`34dd32d6`):** `build_system(..., agent_overlay="")` is
  byte-identical to the 3-arg call → loader/train unaffected; non-empty renders a
  labeled CUSTOMIZATION block last (instruction hierarchy + cache prefix).
  `python -m pytest llm_training/test_system_prompt.py` → 8 passed.
- **Task 10 — `load_skill` (`59b9a0ac`):** `ToolExecutor.execute("<tool>load_skill name=chess-coach</tool>")`
  returns the SKILL.md body; unknown → `error: unknown_skill` (was
  `error: invalid_syntax`).
- **Task 11 — serve via `build_system` (`3a682d41`):** serving system text lists
  `load_skill` + every backend tool + the skills catalog by name+description and
  contains **no** skill body. Removed `skill_prompt()` pre-stuffing + the
  prototype `AGENT_PROMPT`; deleted dead `select_skills`/`skill_prompt`.
- **Task 12 — overlay config (`8a3da5ed`):** `CHESS_AGENT_OVERLAY` (default "")
  → unset means serving system == trained contract.
- **Task 13 — GGUF-only + archive (`487efd9d`):** Ollama fallback removed;
  `backend/model_ollama.py` moved to `legacy [ignore]/dead_backend/` (gitignored,
  untracked); source deletion staged.
- **Task 14 — serve smoke (this commit):** `backend/test_serve_smoke.py` drives
  the real `CoachLoop` + `ToolExecutor` with a scripted model emitting the trained
  shape (lead-in + ONE `<tool>` per step). Found + fixed a parity gap: the loop
  gated on `decision.startswith("<tool>")`, which would treat a lead-in+tool turn
  as a final answer; now it detects the call by search (`"<tool>" in decision`) and
  `normalize_tool_call` closes the tag through a lead-in. Smoke asserts: 3 tools
  execute in order (load_skill→board_state→eval), real chess-coach body returned,
  start-pos board_state/eval correct (no engine needed), final has no XML; a
  dropped-in `SKILL.md` appears in the catalog and is loadable.
- Full backend suite: `pytest backend/test_serve_smoke.py backend/test_serving_system.py
  backend/test_tools_load_skill.py backend/test_colab_config.py` → **15 passed**.

## Next

1. Phase 4 (Task 15): commit/push corpus branch; run `kaggle_e4b_qlora.ipynb` on
   T4 (`--model gemma4_e4b`); export adapter.
2. Phase 4 (Task 16): `eval_routing.py` on held-out + **overlay spot-check** to
   decide Option B (overlay-following SFT) — see Deferred backlog.
3. Phase 5: merge → q4_0 GGUF → serve on the 4060; web smoke (chess routing,
   drop-in SKILL.md, `CHESS_AGENT_OVERLAY` changes tone).
