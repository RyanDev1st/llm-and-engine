# Proof of Concept Results

Generated: 2026-05-13T17:00:02.694128+00:00

## Scope

- Data mode: sample
- Production-valid data: False
- Source rows: 5
- Accepted positions: 5
- Rejected positions: 0
- FEN leakage audit passed: True
- Router training path: linear
- Local-only SFT: False
- Download attempted: False
- Qwen/Gemma status: not_run_linear_baseline
- Trainer blockers: []
- Chess engine is intentionally light: custom linear evaluator from FEN positions using python-chess legal move generation.

## SFT Router Results

- Trainer: linear
- Device: cpu
- CUDA available: True
- CUDA device: NVIDIA GeForce RTX 4060 Laptop GPU
- Router eval examples: 7
- Router tool-name accuracy: 0.857 (6/7)
- Router end-to-end tool-call accuracy: 0.857 (6/7)
- Router macro F1: 0.867
- Review-move argument accuracy: 0.6666666666666666
- Minimum eval support per tool: required=10, passed=False

### Where router succeeds

- best_move: precision=1.000, recall=1.000, f1=1.000, support=2, below minimum=10
- eval: precision=0.667, recall=1.000, f1=0.800, support=2, below minimum=10
- review_move: precision=1.000, recall=0.667, f1=0.800, support=3, below minimum=10

### Why router fails

- tool_confusion:review_move->eval: 1

## Narrator Results

- Narrator eval examples: 3
- Narrator exact accuracy: 0.333
- Narrator grounded rate: 1.000
- Narrator factual accuracy: 0.333

## Chess Engine Training Results

- Legality backend: python-chess
- Training positions: 4
- Eval positions: 1
- Epochs: 10
- Initial heuristic-agreement accuracy: 0.750
- Final heuristic-agreement accuracy: 1.000
- Held-out heuristic-agreement accuracy: 1.000
- Held-out legal prediction rate: 1.000
- Learned weights: `{"bias": 0.0, "capture_value": 0.022222222222222223, "castle": 0.0, "center_control_delta": -0.0125, "gives_check": 0.0, "material_delta": 0.03483999999999998, "mobility_delta": -0.0025000000000000022, "piece_square_delta": 0.013439999999999987, "promotion_value": 0.0}`

## Engine Match Results

- Opponent: same-heuristic static-eval baseline using python-chess legal move generation
- Games: 6
- Engine wins: 3
- Engine losses: 3
- Drawish: 0
- Score rate: 0.500

## Self-Audit

- Generated FEN-blind SFT JSONL from 5 accepted positions with 0 leaked records dropped.
- Trained router path uses linear on cpu; Qwen/Gemma status is not_run_linear_baseline.
- Router eval reports tool_accuracy=0.857, end_to_end_accuracy=0.857, macro_f1=0.867, and review_move argument accuracy=0.6666666666666666.
- Narrator eval reports grounded_rate=1.000 and factual_accuracy=0.333.
- Chess evaluator uses python-chess legal move generation and same-heuristic baseline metrics, not calibrated ELO.
- Production-valid data is False; readiness.production_ready=False with blockers=['llm_trainer_not_used', 'local_transformers_sft_not_completed', 'router_eval_minimum_support_not_met', 'router_end_to_end_accuracy_below_0.95', 'real_kaggle_manifest_not_production_valid', 'stockfish_not_available_for_calibration'].

## Production Readiness

- Production ready: False
- Readiness blockers: ['llm_trainer_not_used', 'local_transformers_sft_not_completed', 'router_eval_minimum_support_not_met', 'router_end_to_end_accuracy_below_0.95', 'real_kaggle_manifest_not_production_valid', 'stockfish_not_available_for_calibration']
- Stockfish available: False
- Stockfish UCI ready: False

## Conclusion

Current artifacts report real measured router, narrator, data, and engine metrics. Production readiness is true only when local Qwen/Gemma SFT, eval support, real Kaggle data, and Stockfish calibration gates all pass.
