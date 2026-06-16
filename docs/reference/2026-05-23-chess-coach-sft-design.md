Parent: none

# Chess Coach SFT + LoRA + Web — Design (v2: execute & complete)

**Date:** 2026-05-23
**Status:** Revised after discovering existing scaffold
**Goal:** Produce a real LoRA adapter on `google/gemma-4-E2B-it` (QLoRA 4-bit; deploy Q4_0 GGUF) for a FEN-blind 9-tool chess coach, then ship the 3-panel website (eval bar | board | chatbox). Source of truth: `chess_assistant_sft_dataset_spec_v3.md`.

## Discovered state (NOT greenfield)
- `src/llm/models/gemma4_e2b/` = real bf16 `Gemma4ForConditionalGeneration` E2B, 3 shards complete. **Training target.** (`gemma4/` is a bigger/compressed sibling — ignore.)
- `src/llm/llm_training/` = capable QLoRA trainer (`run_training`/`TrainConfig`, NF4, paged_adamw_8bit, fused masked loss, vision/audio offload). **Only ever run as 1-step smoke** (`runs/gemma4_lora` = untrained, loss 10.25). Default `model_path` is stale (`src/models/gemma4`). No CLI.
- `src/llm/llm_dataset/` = full validation/audit/replay/dedupe/patch library, **no orchestrator**, never run on the human slices.
- `src/llm/llm_runtime/` = 3-phase loop expecting **JSON router/narrator payloads** — diverges from spec's `<tool>NAME arg=value</tool>` text format. **Stranded; do not use.**
- `src/llm/gemma_chat_site/` = generic stdlib chat UI (loads `src/models/gemma4` via AutoModelForCausalLM, CPU). **Not** the 3-panel chess layout; no board/eval/tool loop. Needs rebuild.
- Pre-checks: E2B shards complete ✓; `<tool>`/`</tool>` survive chat template roundtrip ✓; HF auth ✓ (RyanDev1st); token in HF cache (no env needed).

## Plan (sequential; new code ≤200 lines/file under `src/llm/`; never edit train_cuda.py)
1. **Dataset** — orchestrator script: run audit/schema/mode/routing checks on all 10 human slices; tone-fix flagged cold/robotic narration (keep tool calls/results intact); drop dup C; **author slice D (315)** per §6.1-D; replay-validate vs Stockfish; stratified 90/10 → overwrite `data/sft/chess_assistant_v3_train.jsonl` / `_val.jsonl` (target 3500). Findings report under `legacy/findings/`.
2. **Train** — thin CLI runner → `run_training` on `gemma4_e2b`, fixed data, assistant-only mask. (a) smoke (≤5 steps, prove load+step). (b) **bounded real run** = deliverable: ~500 updates / ≤1 epoch, max_seq 768, r=8→16 if fits, budget ~2–4h. Full 3 epochs = stretch only.
3. **Audit** — routing-accuracy harness on val: per-slice tool-selection accuracy; mode-2 zero `<tool>`; J/K zero tools. Report.
4. **Export** — merge → prebuilt llama.cpp Windows release → `convert_hf_to_gguf` → `Q4_0`. Load-test via llama-cpp-python. Deliverables: adapter dir + `*-Q4_0.gguf`.
5. **Backend** — NEW `src/llm/backend/`: `<tool>` text parser, python-chess board (owns FEN), Stockfish UCI (download official bin to gitignored `runtime/`), 9 tools → spec-exact strings, 3-phase loop (decide→execute→narrate, stop `</tool>`), FastAPI: `/api/chat` (SSE), `/api/state`, `/api/move`, `/api/reset`.
6. **Website** — rebuild `gemma_chat_site/` 3-panel SPA: left eval bar (live engine score), middle chess.com-style interactive board (drag, backend-authoritative), right streaming chatbox. Board + chat drive one backend board; re-sync from `/api/state`.
7. **Verify** — end-to-end manual repro + screenshot.

## Risks / fallbacks
- 8GB VRAM: may force r=8 / seq 768 / batch 1. Smoke de-risks.
- llama.cpp may not yet support gemma4 arch → fallback: serve merged 4-bit via transformers; Q4_0 GGUF stays target deliverable.
- Frontend can be drafted while training runs (parallel).

## Out of scope (spec §9)
explain tool, PGN I/O, multi-game memory, RAG ask_chessbot (canned), DPO, "brilliant".
