# Product path: chess engine backed tool calling

Local proof-of-concept for the FEN-blind chess-coach product idea.

## Run

```bash
python product_demo/chess_tool_demo.py demo
python product_demo/chess_tool_demo.py sft --out product_demo/sample_sft.jsonl
python product_demo/prepare_kaggle_sft.py --input <real_fen_csv> --out-dir product_demo/training_data --data-mode real_kaggle --split-seed 20260513 --eval-ratio 0.2
python product_demo/train_sft_poc.py --train product_demo/training_data/train_sft.jsonl --eval product_demo/training_data/eval_sft.jsonl --out-dir product_demo/poc_models --device cuda --epochs 200 --trainer linear
python product_demo/train_sft_poc.py --train product_demo/training_data/train_sft.jsonl --eval product_demo/training_data/eval_sft.jsonl --out-dir product_demo/poc_models --device cuda --epochs 200 --trainer qwen --model-path <local_qwen_weights> --manifest product_demo/training_data/manifest.json --stockfish-path <optional_stockfish_binary>
python product_demo/train_sft_poc.py --train product_demo/training_data/train_sft.jsonl --eval product_demo/training_data/eval_sft.jsonl --out-dir product_demo/poc_models --device cuda --epochs 200 --trainer gemma4 --model-path <local_gemma4_weights> --manifest product_demo/training_data/manifest.json --stockfish-path <optional_stockfish_binary>
python product_demo/train_chess_engine.py --input <real_fen_csv> --out-dir product_demo/poc_models --epochs 10 --games 6 --max-plies 40
python product_demo/write_poc_results.py --models-dir product_demo/poc_models --out-dir results/poc --manifest product_demo/training_data/manifest.json
python product_demo/evaluate_demo.py --model product_demo/poc_models/router_model.json --engine-model product_demo/poc_models/chess_engine_model.json --eval product_demo/training_data/eval_sft.jsonl --out-dir results/production_eval --games 20 --max-plies 80
python product_demo/web_demo.py
```

Open `http://127.0.0.1:8765` after starting `web_demo.py`.

For the real Kaggle 350k FEN dataset, pass the downloaded CSV path to `--input`; accepted column names are `fen`, `FEN`, `position`, or `Position`.

## What it demonstrates

- Router emits structured tool calls.
- Tool layer owns board state, legal move validation, move review, and evaluation evidence.
- Narrator only explains validated tool results.
- Browser UI calls JSON endpoints backed by the same tool-turn function as the CLI.
- SFT records separate router/tool/narrator turns.
- Kaggle-style FEN CSV conversion creates FEN-blind train/eval JSONL.
- Local router training supports either a torch linear baseline or local-only Qwen/Gemma causal-LM SFT with structured JSON tool-call evaluation.
- SFT training reports parse quality, end-to-end tool-call accuracy, minimum per-tool support, and production blockers instead of claiming hidden readiness.
- Chess-engine training learns custom linear evaluation weights from FEN positions using python-chess legal move generation; current accuracy is heuristic agreement, not ELO strength.
- Production evaluation loads that trained chess-engine artifact for match metrics and replays held-out router prompts on their internal source positions when available.

## Dependencies

```bash
python -m pip install python-chess torch
python -m pip install transformers accelerate  # required only for --trainer qwen or --trainer gemma4 with local weights
```

## Current measured environment

- Python: 3.13.13
- Torch: 2.6.0+cu124
- CUDA: available
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Legal move backend: python-chess 1.11.2
- Missing external executables: Kaggle CLI, Stockfish

## Scope

This is a measured local proof-of-concept, not calibrated ELO production chess service. It no longer uses the old hand-written legal-move engine for training/evaluation; legality comes from python-chess. Full production delivery still needs the real 350k FEN CSV and an installed Stockfish/UCI engine for calibrated strength metrics.

Current limitations:
- Kaggle CLI is not installed, so this environment used the available CSV path only.
- Stockfish executable is not installed, so engine strength is not calibrated ELO unless `--stockfish-path` or `stockfish` on PATH passes UCI smoke test.
- Narration model is basic template retrieval; current factual accuracy is reported separately from grounded rate.
- Qwen/Gemma router training now requires local weights and local Python dependencies; no hidden model download is attempted.
- `train_router.py` is the older Naive Bayes router experiment; the supported proof-of-concept path uses `train_sft_poc.py` and `poc_models/router_model.json`.
