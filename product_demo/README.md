# Product demo: chess engine backed tool calling

Minimal local demo for the FEN-blind chess-coach product idea.

## Run

```bash
python product_demo/chess_tool_demo.py demo
python product_demo/chess_tool_demo.py sft --out product_demo/sample_sft.jsonl
python product_demo/prepare_kaggle_sft.py --input product_demo/sample_kaggle_fens.csv --out-dir product_demo/training_data
python product_demo/train_router.py --train product_demo/training_data/train_sft.jsonl --eval product_demo/training_data/eval_sft.jsonl --out product_demo/trained_router.json
python product_demo/web_demo.py
```

Open `http://127.0.0.1:8765` after starting `web_demo.py`.

For the real Kaggle 350k FEN dataset, pass the downloaded CSV path to `--input`; accepted column names are `fen`, `FEN`, `position`, or `Position`.

## What it demonstrates

- Router emits structured tool calls.
- Tool layer owns mutable board state, legal move validation, move review, and evaluation evidence.
- Narrator only explains validated tool results.
- Browser UI calls JSON endpoints backed by the same tool-turn function as the CLI.
- SFT sample records separate router/tool/narrator turns.
- Kaggle-style FEN CSV conversion creates FEN-blind train/eval JSONL.
- Local router training learns tool selection from generated SFT records and reports held-out accuracy.

## Scope

This is product-demo code, not production chess logic. It uses a tiny deterministic material/mobility oracle so the demo runs without Stockfish or python-chess. Replace `demo-material-mobility-v1` with python-chess plus Stockfish for paid pilot delivery.

Current limitations:
- no castling, en passant, checkmate, stalemate, or draw rules
- no SAN parser; demo accepts UCI moves only
- no real Stockfish search or NNUE evaluation
- no neural fine-tuning; `train_router.py` is a local classifier proving the training/eval loop shape
