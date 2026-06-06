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

## Next action

Start Phase 1, Task 1: add `build_system()` to `src/llm/llm_training/system_prompt.py` + test. Everything downstream depends on the shared train==serve contract renderer.
