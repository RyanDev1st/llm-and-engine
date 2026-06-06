# Handoff: chess-coach agent — Kaggle-train / local-host

## The goal (single path)

Train a Gemma-4 chess-coach **agent** (tool-router/narrator, not a chess engine) as a **LoRA adapter** via QLoRA on **Kaggle T4**, then serve it **locally as q4_0 GGUF on the RTX 4060 (8GB)**.

- **Model:** Gemma 4 **E4B** preferred; **E2B** fallback only if E4B QLoRA OOMs on T4 or local E4B serving is too slow.
- **Why this split:** training holds weights+grads+optimizer+activations (~2–3× inference). E4B QLoRA (~9–12GB) fits a T4's 16GB but NOT the local 8GB. E4B q4_0 inference (~4.5GB, measured) DOES fit the 4060.
- Plan of record: `implementation.md`. Decision memory: `chess-agent-train-host-split`.
- FPT/Qwen path is **abandoned** → `legacy [ignore]/archived_plans/implementation_fpt.md`.

## Current blocker (do this first)

The v1.2 corpus is **not trainable as-is**. QC (python-chess, 2026-06-06):
- 59% of `move` rows are **illegal** for their `position_fen`; the `tool` turn fabricates `success: e4`.
- Only `e4` is ever played (monoculture); 93% of val final-targets leak from train; ~40% finals carry banned persona openers.
- Routing scaffolding is excellent (100% `load_skill`-first, 655 distinct tool sequences) — keep it.

Root cause: `src/llm/llm_dataset/v1/renderer/chess.py` hardcodes `move san=e4` / `success: e4` / `turn=white`; `validate.py`'s `engine_grounded` never checks legality. Memory: `chess-sft-v1_2-illegal-move-bug`.

## Plan phases (see implementation.md for full TDD tasks)

1. **Phase 1 — fix the generator (BLOCKER):** FEN-grounded `board_facts.py`, rewrite `renderer/chess.py` to emit legal moves + real tool echo, add validator legality gate, flatten persona openers, dedup val finals, regenerate v1.2, QC gate (0 illegal, <1% leak, 0 banned openers).
2. **Phase 2 — train on Kaggle T4:** add `--model` flag (E4B/E2B) to `run_train.py`, author `kaggle_e4b_qlora.ipynb`, produce LoRA adapter, run routing audit.
3. **Phase 3 — serve locally:** `export_gguf.py` (merge adapter → q4_0 GGUF), point `CHESS_GGUF_PATH` at it, smoke the web app on the 4060.

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

Start Phase 1, Task 1 (`board_facts.py` + tests). Everything downstream depends on a grounded corpus.
