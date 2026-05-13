# Product path: chess engine backed tool calling

Local proof-of-concept for the FEN-blind chess-coach product idea.

## Run

```bash
python product_demo/chess_tool_demo.py demo
python product_demo/chess_tool_demo.py sft --out product_demo/sample_sft.jsonl
python product_demo/prepare_kaggle_sft.py --input <real_fen_csv> --out-dir product_demo/training_data
python product_demo/train_sft_poc.py --train product_demo/training_data/train_sft.jsonl --eval product_demo/training_data/eval_sft.jsonl --out-dir product_demo/poc_models --device cuda --epochs 200
python product_demo/train_chess_engine.py --input <real_fen_csv> --out-dir product_demo/poc_models --epochs 10 --games 6 --max-plies 40
python product_demo/write_poc_results.py --models-dir product_demo/poc_models --out-dir results/poc
python product_demo/evaluate_demo.py --model product_demo/poc_models/router_model.json --engine-model product_demo/poc_models/chess_engine_model.json --eval product_demo/training_data/eval_sft.jsonl --out-dir results/production_eval
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
- Local router training learns tool selection from generated SFT records and reports held-out accuracy.
- SFT training uses a basic torch linear router on CUDA when available, plus a grounded narrator component.
- Chess-engine training learns custom linear evaluation weights from FEN positions using python-chess legal move generation.
- Production evaluation loads that trained chess-engine artifact for match metrics.

## Dependencies

```bash
python -m pip install python-chess torch
```

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
- Stockfish executable is not installed, so engine strength is not calibrated ELO.
- Narration model is basic template retrieval; router is basic torch linear classifier.
- Metrics are real measured outputs from local runs, but sample-data metrics are not generalization claims.
