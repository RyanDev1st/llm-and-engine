# Handoff: chess-coach agent — Kaggle-train / local-host

## The goal (single path)

Train a Gemma-4 chess-coach **agent** (tool-router/narrator, not a chess engine) as a **LoRA adapter** via QLoRA on **Kaggle T4**, then serve it **locally as q4_0 GGUF on the RTX 4060 (8GB)**.

- **Model:** Gemma 4 **E4B** preferred; **E2B** fallback only if E4B QLoRA OOMs on T4 or local E4B serving is too slow.
- **Why this split:** training holds weights+grads+optimizer+activations (~2–3× inference). E4B QLoRA (~9–12GB) fits a T4's 16GB but NOT the local 8GB. E4B q4_0 inference (~4.5GB, measured) DOES fit the 4060.
- Plan of record: `implementation.md`. Decision memory: `chess-agent-train-host-split`.
- FPT/Qwen path is **abandoned** → `legacy [ignore]/archived_plans/implementation_fpt.md`.

## Contract decision (2026-06-06): Option B — agentic harness

`load_skill` is intended. The agent is an agentic harness: **primary** = operate the chess environment/tools; **secondary** = dynamically load any user-provided `SKILL.md`. Full audit: `docs/2026-06-06-v1.2-dataset-alignment-audit.md`. Plan of record: `implementation.md` (5 phases).

## Current blocker (do this first)

The v1.2 corpus is **not trainable as-is** — two layers of bugs:

**Contract (alignment):** both loaders inject a fixed 9-tool `SYSTEM_PROMPT` and serialize ONLY `messages`, so `skills_index`/`tool_manifest`/`plugin_context` are discarded and `load_skill`/`board_state` are undeclared (100% rows open with `load_skill`; 43% of tool calls hit non-declared tools; 65k Mode-2 chains the prompt forbids). The backend has no `load_skill` tool.

**Content:** 59% illegal moves + fabricated `success: e4`; e4 monoculture; board_state turn wrong in 12.6k rows + embeds fen in "basic"; 93% val leak; 65% persona openers. Root cause: `renderer/chess.py` hardcodes; `validate.py` never checks legality.

Keep: 100% `load_skill`-first structure, 653 distinct tool-sequence shapes, balanced 11-pattern reject pool.

## Plan phases (see implementation.md for full TDD tasks)

1. **Harness contract wiring:** shared `build_system(skills_index, tool_manifest, plugin_context)` in `system_prompt.py`; loaders compose the per-row system from the envelope; loader test that every called tool is declared.
2. **Content correctness:** `board_facts.py` (legal moves + real echoes), rewrite `renderer/chess.py`, validator legality gate, flatten personas, de-leak split, regenerate, QC gate.
3. **Backend harness parity:** backend `load_skill` tool + inject skills/manifest via the SAME `build_system()` (train == serve); archive dead `model_ollama.py`.
4. **Train E4B QLoRA on Kaggle T4** → adapter → routing audit (`kaggle_e4b_qlora.ipynb` ready, `run_train --model` ready).
5. **Serve locally:** merge adapter → q4_0 GGUF → web app smoke on the 4060.

## Live pipeline facts

- Generator: `python -m llm_dataset.v1.generate --profile v1.2` → `data/sft/v1_2/{accepted,rejected}.jsonl`; then `python -m llm_dataset.v1.build --profile v1.2` → `data/sft/v1_2_{train,val}.jsonl`.
- Trainer: `python -m llm_training.run_train` (reads v1_2_train/val; base currently hardcoded `gemma4_e2b` → Task 7 parametrizes).
- Backend serving already supports `CHESS_GGUF_PATH` (`src/llm/backend/model_gguf.py`).
- Stress tests this session: `src/llm/llm_training/stress_test_gemma4.py` (HF/nf4) and `stress_test_gemma4_gguf.py` (llama.cpp). Note: local `llama_cpp` 0.3.23 is a **CPU build** (GPU offload needs a CUDA rebuild).

## Constraints

- Never edit/import `legacy [ignore]/` (now gitignored).
- No secrets in code/logs/commits; Kaggle HF token via Kaggle Secrets.
- Stage intended files only; no `git add -A`.
- `*.gguf` / `*.safetensors` are gitignored — commit code, not weights.
- Watch long-running shells; clean GPU memory between heavy runs.

## Progress (2026-06-06) — data quality first

- **Phase 1 DONE** (harness contract wiring), TDD, committed:
  - `system_prompt.build_system(skills_index, tool_manifest, plugin_context)` — shared train==serve renderer; `BASE_HARNESS` allows skill-first multi-step.
  - Both loaders compose the per-row system from the envelope; loader-contract test asserts every called tool is declared.
- **Phase 2 DONE (code), regen running:** all committed, TDD:
  - T3 `board_facts.py` (FEN→legal move, real echoes mirroring `backend/game.py`/`tools.py`).
  - T4 `renderer/chess.py` grounded — legal+diverse moves, real tool results, correct side-to-move.
  - T5 `validate.py` legality gate (`illegal_move` + `board_state_grounded`).
  - T6 `tone.py` persona openers removed (tools not tone).
  - T7 `build.split_train_val()` de-leak (no val final in train).
  - Fix: slice B `legal_moves square=<sq>` grounded (manifest requires square).
  - **Real-Stockfish smoke: 66 rows across all slices → 0 validate violations, 0 illegal, diverse moves, 0 personas.**
- **T8 IN PROGRESS:** full regen running in background → `data/sft/v1_2/{accepted,rejected}.jsonl` + `build` split. Log: `build/regen_v1_2.log` (Stockfish present at `src/llm/runtime/stockfish/...`). Overwrites the old broken corpus (recoverable in git history).

## Phase 2 DONE — corpus regenerated + QC green (committed)

Regenerated `data/sft/v1_2`; QC gate all green: 0 illegal moves (was 59%), 582 distinct moves (was 1), 0 board_state turn mismatch (was 12,621), 0 undeclared tool calls (was 47k), 0.0% val leak (was 93%), 0 persona openers (was 65%), 100% load_skill-first. Build ran `assert_valid` on all 50,002 accepted rows. Counts: 49,638 train / 364 val; 39,053 harness_chess + 10,585 universality. NOTE: `v1_2_train.jsonl` is ~283MB (manifest repeated per row) — mind GitHub push limits.

## Open quality nuance — val is small (364)

De-leak moved ~4,600 val rows into train because finals are move-parameterized templates that collide (e.g. "Played g5..." x175). Leak-safe but val is small/biased. **Recommended fix:** split by intent/scenario family (hold out whole scenarios ~10%) in `build.split_train_val` instead of dropping colliding finals.

## Conversational-agent enrichment (design approved 2026-06-06)

Goal: agent converses like a coding agent — lead-in narration, tool, narrate, final + guiding question. Design: `docs/2026-06-06-conversational-agent-design.md`. Decisions: **backend auto-save** (not a model tool); **one lead-in sentence before each tool**.

- **Foundation DONE (committed, TDD):** `validate.py` accepts `lead-in + one <tool>` per assistant message (search-based), `one_tool_per_message` rule (exactly one tool per inference step), final = last no-tool assistant turn; `BASE_HARNESS` documents lead-in + guiding-question finals.
- **Still to do (the data shape):**
  1. `renderer/chess.py` + `renderer/universality.py`: emit a short lead-in before each tool call; coaching finals = brief grounded assessment + one guiding question; load_skill result = a real multi-line SKILL.md body; sometimes load a non-chess-coach skill (dynamic-SKILL.md objective).
  2. Regenerate v1.2 + QC gate (existing checks + new: each action turn has exactly one tool; coaching finals end with a question).
  3. Backend (Phase 3): serving loop streams lead-in → runs tool on `</tool>` → continues; auto-save each turn to disk; implement `load_skill` returning the body.

## Open quality nuance — val is small (364)

De-leak moved leaked rows to train (finals are templated). Leak-safe but small/biased. Recommended fix: split by intent/scenario family in `build.split_train_val`.

## Final skill/tool contract (2026-06-06, refined 2026-06-07) — see chess-agent-skill-tool-contract memory

- `load_skill` is a TOOL; skill = text/context (progressive disclosure: catalog always in context, body via load_skill, persists). No `<skill>` tag.
- **One tool call per inference step** (per assistant message), MANY across the agentic loop — like a coding agent (act → read result → act). `one_tool_per_message` rule in `validate.py`; `BASE_HARNESS` says "EXACTLY ONE call per step". (This REVERSED the earlier "multiple per turn" reading; keep no_exact_duplicate + max_six_tool_calls.)
- Cross-domain skill diversity: index offered ~2,737 skills but the agent LOADED only 2. **FIXED** — see below.

## Cross-domain skill routing — DONE (code + smokes 2026-06-07), regen running

The renderer no longer only loads chess-coach. New capability, TDD, committed:
- `domains.py`: 8 real domains (code/math/writing/cooking/data/fitness/travel/resume), each a real multi-line SKILL.md body + one domain tool; `synthetic_domain()` mints freshly-named skills over 20 topics so the model must route by DESCRIPTION, not names; `pick_domain()` = 40% real / 60% synthetic.
- `renderer/skill_routing.py`: load fitting skill → read body → call its tool → guiding-question final. One tool/step; lead-in before each; `normalize=True` loads hood-human-chat first (two skills across separate steps).
- `generate.py`: slice `V1_O_cross_domain_skill_routing` wired in (base 70). `audit.py`: `loaded_skill_diversity` metric + gate (≥50).
- `renderer/leadins.py` + edits to `chess.py`/`universality.py`: lead-in before every tool; coaching finals end with a guiding question; envelopes now regex-based.
- Smokes: routing-only (120 rows, diversity 80, 0 fails); mixed chess+routing (150 rows, 0 fails, 60/60 coaching finals end with "?").
- **REGEN LANDED (2026-06-07, committed `8572d7ed`):** full regen seed 20260525 → build → audit `freeze_ok=True`, **0 failures**. `loaded_skill_diversity = 447` (was 2). accepted 50,060 / rejected 7,500; train 49,469 / val 591. synthetic_share 0.326, generic_final 0.000, reject diversity 11, V1_O = 749. (One earlier near-miss — accepted 49,990 < 50k from rounding — fixed in `plan_for_profile` top-up, `3b5b904b`.)

## Next action

1. **Phase 3 backend parity** (do first; NOTE: backend files engine.py/inference.py/state_api.py/tools.py have UNCOMMITTED working-tree edits from a prior session + untracked `backend/skills.py` — review/triage those before editing). `ToolExecutor._dispatch` has NO `load_skill` branch (unknown→`error: invalid_syntax`); `skills.py` (`load_skills`/`select_skills`) exists but isn't wired into the executor or the serving system prompt. Add `load_skill` returning the body; build the serving system via the SAME `build_system(skills_index, tool_manifest, plugin_context)`; auto-save each turn; stream lead-in → run tool on `</tool>` → continue; archive dead `backend/model_ollama.py`.
2. Phase 4 Kaggle E4B QLoRA → adapter (`kaggle_e4b_qlora.ipynb`, `run_train --model gemma4_e4b`). Phase 5 merge → q4_0 GGUF → local serve + web smoke.
3. Optional: val is small (591) — de-leak drops templated-final collisions. Split by intent/scenario family in `build.split_train_val` for a larger leak-free val.
