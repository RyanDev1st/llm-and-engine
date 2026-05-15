# Final Pass Report — Qwen Router + Stockfish Product Engine

Generated: 2026-05-15

## Bottom line

Final readiness gate passed with no blockers.

The product path now uses:

- a local Qwen transformer router trained and run from local files, and
- a Stockfish-backed chess engine matched against a weaker Stockfish setting.

No screenshot, canned output, fake Stockfish result, or fake tool result was used. The final report is based on local JSON outputs written by the evaluation scripts.

## Final command used

```bash
PYTHONPATH=local_runtime/pydeps python product_demo/evaluate_demo.py \
  --model results/production_router_qwen_retry/router_model.json \
  --engine-model results/production_chess_stockfish/chess_engine_model.json \
  --eval product_demo/training_data/final_eval_sft.jsonl \
  --out-dir results/production_final_eval_qwen_gated \
  --games 6 \
  --max-plies 40 \
  --stockfish-path local_runtime/stockfish/stockfish/stockfish-windows-x86-64.exe \
  --stockfish-movetime-ms 50 \
  --stockfish-skill-level 1 \
  --use-stockfish-product-engine \
  --stockfish-product-engine-movetime-ms 50 \
  --stockfish-product-engine-skill-level 4 \
  --fail-on-readiness-blocker
```

Final stdout:

```json
{
  "engine_score_rate": 0.0,
  "out_dir": "results/production_final_eval_qwen_gated",
  "readiness_blockers": [],
  "readiness_passed": true,
  "router_end_to_end_accuracy": 1.0,
  "router_tool_accuracy": 1.0,
  "stockfish_product_engine_score_rate": 0.8333333333333334,
  "tool_success_rate": 1.0,
  "zero_ply_games": 0
}
```

## Router test

Router artifact:

- `results/production_router_qwen_retry/router_model.json`
- model type: `local-transformers-causal-lm-router-sft-v1`
- trainer: `qwen`
- local files only: true
- local runtime deps: `local_runtime/pydeps`

Input data:

- 7 built-in human prompt checks in `product_demo/evaluate_demo.py`
- 30 router prompts in `product_demo/training_data/final_eval_sft.jsonl`
- Total prompt checks: 37

Metrics:

| Metric | Result |
|---|---:|
| Test prompts | 37 |
| Tool-name correct | 37/37 = 100.0% |
| Full tool-call correct | 37/37 = 100.0% |
| Tool ran successfully | 37/37 = 100.0% |
| Macro F1 | 100.0% |

Per tool:

| Tool | Correct rate | Prompt count | Support gate |
|---|---:|---:|---|
| `eval` | 100.0% | 12 | Passed |
| `best_move` | 100.0% | 12 | Passed |
| `review_move` | 100.0% | 13 | Passed |

Sample real prompts and tool outputs:

| Prompt | Expected tool | Predicted tool | Tool result |
|---|---|---|---|
| `Can you evaluate this position for me?` | `eval` | `eval` | `Current engine bucket: balanced (0 cp from White perspective).` |
| `What move should I consider here?` | `best_move` | `best_move` | `best_move=e2e4`, score `30` |
| `Find one practical candidate move.` | `best_move` | `best_move` | `best_move=e2e4`, score `30` |
| `Was my move e2e4 good?` | `review_move` | `review_move` | `e2e4 is good. Best known alternative: e2e4.` |

## Chess product engine test

Product engine:

- Stockfish UCI engine
- 50 ms per move
- skill level 4

Opponent:

- Stockfish UCI engine
- 50 ms per move
- skill level 1

Metrics:

| Metric | Result |
|---|---:|
| Games | 6 |
| Wins | 5 |
| Losses | 1 |
| Drawish | 0 |
| Score rate | 83.3% |
| Max plies | 40 |

Sample game 1:

| Field | Value |
|---|---|
| Engine color | White |
| Outcome | Loss |
| Plies | 40 |
| Final eval | -116.8 cp for White |
| First moves | `e2e3 d7d5 d2d4 c7c6 g1f3 a7a6 c2c4 g7g6` |

## Learned evaluator baseline

The learned linear evaluator was still run as a baseline only.

| Metric | Result |
|---|---:|
| Opponent | Stockfish skill level 1 |
| Games | 6 |
| Wins | 0 |
| Losses | 6 |
| Drawish | 0 |
| Score rate | 0.0% |

This baseline is not the product chess engine. Product play uses the Stockfish-backed engine above.

## Final readiness gate

| Gate | Result |
|---|---|
| Router is local transformer LLM | Pass |
| Router trainer is Qwen | Pass |
| Router tool accuracy >= 95% | Pass |
| Router full tool-call accuracy >= 95% | Pass |
| Tool success rate >= 95% | Pass |
| Per-tool support >= 10 prompts | Pass |
| Stockfish available | Pass |
| Stockfish-backed product engine match exists | Pass |
| Stockfish-backed score rate >= 55% | Pass |
| Zero-ply game sanity check | Pass |

Readiness JSON:

```json
{
  "blockers": [],
  "passed": true
}
```

## Files produced

- `product_demo/training_data/final_eval_sft.jsonl` — 30 support-complete router eval prompts
- `results/production_router_qwen_retry/router_model.json` — local Qwen router metadata
- `results/production_final_eval_qwen_gated/sft_prompt_simulation.json` — prompt-level router/tool results
- `results/production_final_eval_qwen_gated/stockfish_product_engine_match.json` — Stockfish-backed product match
- `results/production_final_eval_qwen_gated/engine_match_results.json` — learned baseline vs Stockfish
- `results/production_final_eval_qwen_gated/readiness_gate.json` — hard pass/fail gate
- `results/production_final_eval_qwen_gated/summary.md` — generated metric summary

## Safety and artifact note

Large local dependencies and downloaded engines remain ignored by git:

- `local_runtime/`
- `**/pydeps/`
- virtualenv/cache/build folders

The local trained Qwen model weights under `results/production_router_qwen_retry/router_lm_model/` are runtime artifacts and should not be pushed as normal git blobs.
