Parent: none

# Chess Coach SFT + LoRA + Web — Design

**Date:** 2026-05-23
**Status:** Draft for user review
**Goal:** Fine-tune `google/gemma-4-E2B-it` (LoRA, QLoRA 4-bit; deploy as Q4_0 GGUF) into a FEN-blind chess coach that routes user chat to one of 9 tools against a real Stockfish backend, narrates results warmly, and ships behind a 3-panel website (eval bar | interactive board | chatbox).

Driven by `chess_assistant_sft_dataset_spec_v3.md` (the dataset/tool/system-prompt contract — authoritative).

---

## Layout (all new code under `src/llm/`)

| Path | Purpose |
| --- | --- |
| `src/llm/data_tools/` | Audit, tone-fix, slice-D authoring, dedupe, replay-validate, split. |
| `src/llm/trainer/` | QLoRA train (peft+bnb), chat-template formatting, label masking, eval harness. |
| `src/llm/backend/` | python-chess + Stockfish UCI, 9-tool executor, 3-phase inference loop, FastAPI. |
| `src/llm/web/` | 3-panel SPA (static HTML/CSS/JS) served by the backend. |
| `src/llm/runtime/` | gitignored: Stockfish bin, base/merged weights, GGUF, llama.cpp release. |

**Rule:** every `.py`/`.js` source file ≤200 lines (CLAUDE.md hard cap; split into more files in the same folder). HTML/CSS/data files exempt. `runtime/` is gitignored (large binaries/weights). Add a `src/llm/` row to CLAUDE.md repo map in the same change set.

---

## Subsystem 1 — Dataset

**Inputs:** human slices in `data/sft/slices/slices/` (A,C,E,F,G) and `data/sft/slice/slice/` (B,C,H,I,J,K). Counts already match spec targets. Missing: **D (eval, 315)**. **C duplicated** (two files, different id schemes/tone).

**Steps:**
1. **Audit** (`audit.py`): per record run §7 checks — schema, legal role sequence, tool grammar `^<tool>(\w+)((?:\s+\w+=\S+)*)</tool>$`, mode-discipline (no `<tool>` in assistant-after-tool), slice-routing sanity (J/K zero tools; A–I expected tool family). Emit a JSON report of violations.
2. **Tone pass** (`tone_fix.py`): flag assistant narration that is cold/robotic/lecturing/over-long (>3 sentences, imperative "Please specify…", no warmth). Rewrite to warm + friendly, 1–3 sentences, **without** changing tool calls or tool-result strings. Manual-reviewable diff written to findings.
3. **Dedupe C:** keep the higher-quality C file (better tone, valid grammar), drop the other.
4. **Author slice D** (`gen_slice_d.py` output → `data/sft/slice/slice/slice_D_315.json`): 315 implicit-eval convos per §6.1-D. ~15% mate scores, ~15% timeout/engine_unavailable stress, aggressive phrasing variety. System prompt verbatim from spec §3.
5. **Replay-validate** (`replay.py`): real `chess.Board()` + Stockfish. For move/undo/legal_moves/list_pieces exact-match; eval/best_move/review_move/threats numeric within ±0.30 pawns or same sign+magnitude class, mate exact. ask_chessbot skipped.
6. **Assemble** (`assemble.py`): merge all slices, dedupe near-identical user turns (>0.85), stratified 90/10 split → overwrite `data/sft/chess_assistant_v3_train.jsonl` / `_val.jsonl`. Target 3500.

**Verify:** validator first-pass ≥95%; replay-check passes; final counts logged per slice.

---

## Subsystem 2 — Training (QLoRA)

**Engine:** peft + bitsandbytes (already installed) + transformers `Trainer`; `pip install datasets`. (unsloth/trl rejected: install/version risk on Py3.13 + transformers 5.8.)

**Base:** `google/gemma-4-E2B-it`, loaded 4-bit NF4 (double-quant, bf16 compute). Gated — requires `HF_TOKEN` in env (user provides; never in chat/repo).

**Formatting:** apply Gemma chat template to each conversation. **Label masking:** loss only on assistant tokens (tool calls + narration); mask system/user/tool tokens. This teaches routing + glue, not user/tool text.

**Hyperparams (starting point):** LoRA r=16, alpha=32, dropout=0.05, target `q,k,v,o,gate,up,down` proj; lr 2e-4 cosine, warmup 3%, grad-accum to ~16 effective batch, grad checkpointing, max_seq ~1024, 3 epochs. Tune to fit 8GB.

**Runs:** (a) **smoke** ~50 steps — proves pipeline, no OOM, loss finite. (b) **full** ~3 epochs.

**Verify — routing eval harness** (`eval_routing.py`) on val set:
- Mode-1 tool-selection accuracy per slice (correct tool family).
- Mode-2 discipline: assistant-after-tool emits **zero** `<tool>`.
- J/K negative routing: zero tools.
- Report confusion matrix + per-slice accuracy. Target: high routing accuracy, zero mode-2 violations.

---

## Subsystem 3 — Export

1. Merge LoRA into bf16 base (`merge.py`).
2. `convert_hf_to_gguf.py` → GGUF (from prebuilt **llama.cpp Windows release**, no C++ build).
3. `llama-quantize` → **Q4_0**.
4. Deliverables: LoRA adapter dir + `gemma-4-E2B-chesscoach-Q4_0.gguf`.

**Verify:** load Q4_0 via `llama-cpp-python` (prebuilt CUDA wheel), sample one prompt per slice intent; outputs well-formed.

---

## Subsystem 4 — Backend + Website

**Backend** (`src/llm/backend/`):
- `engine.py`: Stockfish UCI wrapper (download official Windows binary into `runtime/`).
- `board_state.py`: python-chess board, move stack, FEN owned here (model never sees it).
- `tools.py`: 9 tools → spec-exact return strings incl. all `move` shapes + universal errors; hard 5s timeout.
- `inference.py`: 3-phase loop (decide → execute tool → narrate), same system prompt all phases; model via llama-cpp-python (Q4_0 + our adapter or merged GGUF).
- `server.py`: FastAPI — `POST /api/chat` (SSE stream), `GET /api/state` (FEN, eval, legal, history), `POST /api/move`, `POST /api/reset`. Serves `web/` static.

**Website** (`src/llm/web/`): single-page, 3-panel.
- **Left:** vertical eval bar, live engine score (white/black advantage), numeric readout.
- **Middle:** chess.com-style interactive board (drag-drop pieces; backend authoritative for legality; highlights, last-move, check). Library: chessground or chessboard-element + board render; chess.js only for client-side UX hints.
- **Right:** standard LLM chatbox — message history, streaming tokens, typing indicator, user input, tool-call activity shown subtly.
- Board moves and chat both mutate the one backend board; eval bar + board re-sync from `/api/state` after every action.

**Verify:** launch server, manual end-to-end repro (play moves by drag + by chat; ask "how's my position" → eval routes, bar updates; ambiguous move → clarify loop). Screenshot.

---

## Risks
- Gemma-4 E-series 4-bit + LoRA target-module names may differ from Gemma-3; confirm module names at load time before full run.
- 8GB VRAM: may need r=8 / max_seq 768 / batch 1 to fit. Smoke run de-risks.
- llama.cpp converter must support Gemma-4 arch; if not, fall back to serving merged HF model 4-bit via transformers in backend (still 4-bit, GGUF deferred). Q4_0 GGUF remains the target deliverable.
- HF gating: requires accepted license + token.

## Out of scope (per spec §9)
explain tool, PGN import/export, multi-game memory, RAG ask_chessbot (canned answers), DPO, "brilliant" classification.

## Verification summary
1. Data: validator ≥95%, replay pass, 3500 count.
2. Train: loss converges; routing harness high accuracy, zero mode-2 violations.
3. Export: Q4_0 loads, samples well-formed.
4. App: end-to-end manual repro + screenshot.
