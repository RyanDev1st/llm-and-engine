# Proof of Concept Results

Generated: 2026-05-13T11:47:33.203123+00:00

## Scope

- Basic local models only; no high-end model import.
- SFT trained as CUDA/CPU linear router plus narrator template model from generated FEN-blind JSONL.
- Chess engine trained as custom linear evaluator from FEN positions using python-chess legal move generation.
- Progress-based artifacts saved as JSON under this folder.

## SFT Training Results

- Device: cuda
- CUDA available: True
- CUDA device: NVIDIA GeForce RTX 4060 Laptop GPU
- Router eval examples: 3
- Router accuracy: 1.000 (3/3)
- Narrator eval examples: 3
- Narrator exact accuracy: 0.333
- Narrator grounded rate: 1.000

## Chess Engine Training Results

- Legality backend: python-chess
- Training positions: 4
- Eval positions: 1
- Epochs: 5
- Initial training accuracy: 0.750
- Final training accuracy: 1.000
- Held-out eval accuracy: 1.000
- Held-out legal prediction rate: 1.000
- Learned weights: `{"bias": 0.0, "capture_value": 0.022222222222222223, "castle": 0.0, "center_control_delta": -0.0125, "gives_check": 0.0, "material_delta": 0.03483999999999998, "mobility_delta": -0.0025000000000000022, "piece_square_delta": 0.013439999999999987, "promotion_value": 0.0}`

## Engine Match Results

- Opponent: static-eval baseline using python-chess legal move generation
- Games: 6
- Engine wins: 3
- Engine losses: 3
- Drawish: 0
- Score rate: 0.500

## Self-Audit

- Implemented local training loop: generated SFT JSONL, trained CUDA router, trained narrator, trained chess evaluator, wrote measured metrics.
- No high-end model imported; models are torch linear router, template narration, and linear move evaluator.
- FEN remains internal; SFT visible prompts/narrations do not expose raw FEN.
- Chess legality now uses python-chess, including castling, en passant, terminal rules, and legal move generation.
- Kaggle CLI and Stockfish executable are not installed; current metrics use the available sample FEN CSV and static-eval baseline, not calibrated ELO.

## Conclusion

Current product path uses python-chess legality and measured CUDA/router/chess metrics. It is still not calibrated ELO because Stockfish/UCI and full Kaggle data are not installed in this environment.
