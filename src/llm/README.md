# Chess Coach (Gemma-4 E2B LoRA) — `src/llm`

A FEN-blind chess-coach LLM: it routes user chat to one of 9 tools against a real
Stockfish backend and narrates the results warmly. Trained with QLoRA on
`google/gemma-4-E2B-it`, served behind a 3-panel web app.

Spec of record: [`../../chess_assistant_sft_dataset_spec_v3.md`](../../chess_assistant_sft_dataset_spec_v3.md).

## Layout

| Path | Purpose |
| --- | --- |
| `llm_dataset/build/` | v3 dataset build: clean, author slice D, augment J/K, assemble train/val. |
| `llm_dataset/` (other) | Pre-existing validation/contract library (format-agnostic helpers reused). |
| `llm_training/` | QLoRA trainer (`run_train`), routing audit (`eval_routing`), GGUF export (`export_gguf`). |
| `backend/` | python-chess `Game`, Stockfish `Engine`, 9-tool executor (`tools`), 3-phase loop (`inference`), HF model backend, stdlib `server`. |
| `gemma_chat_site/` | 3-panel web app (eval bar · interactive board · streaming chat). |
| `models/`, `runtime/` | Base weights + Stockfish (gitignored). |
| `llm_runtime/` | Stranded prior work (JSON tool format) — not used; backend follows the spec's `<tool>` text format. |

## Pipeline

```bash
# 1. Build the dataset (cleans human slices, authors slice D, assembles train/val)
cd src/llm && python -m llm_dataset.build.assemble

# 2. Train (run from src/llm). Smoke first, then full.
python -m llm_training.run_train --smoke --targets attn-only --max-seq 1152
python -m llm_training.run_train --targets attn-only --max-seq 1152 --epochs 3

# 3. Audit routing accuracy on the val set
python -m llm_training.eval_routing runs/gemma4_chess

# 4. Export Q4_0 GGUF (optional; web app also serves 4-bit + adapter directly)
python -m llm_training.export_gguf runs/gemma4_chess

# 5. Serve the web app (from src/llm), passing the trained adapter dir
python -m backend.server ../../runs/gemma4_chess        # http://127.0.0.1:7860
```

## Notes

- 8 GB VRAM: train in 4-bit NF4, LoRA on attention projections (`attn-only`), seq 1152.
- The board is authoritative: drag-moves and chat tool calls mutate the one game;
  the frontend re-syncs from `/api/state` (FEN, eval bar, legal moves, history).
- Tool returns follow spec §2 exactly; `ask_chessbot` uses a canned KB (RAG is v2).
