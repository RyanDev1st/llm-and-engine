# Final Pass Report — Real Local Evaluation

Generated: 2026-05-15

## Bottom line

The router/tool-calling path now passes the local proof-of-concept test set. The chess work now includes real Stockfish runs from an isolated local runtime.

This is still not a true production-grade chess engine: the learned linear chess evaluator lost all 6 games against Stockfish. A Stockfish-backed product engine did beat a weaker Stockfish setting in the same harness, but that proves the harness and integration, not a new trained engine.

No staged screenshots, canned output, fake Stockfish result, or fake tool result was used.

## What was tested

### Tool router

Command used:

```bash
python product_demo/evaluate_demo.py --model results/production_router_linear/router_model.json --engine-model results/production_chess_stockfish/chess_engine_model.json --eval product_demo/training_data/eval_sft.jsonl --out-dir results/production_final_eval --games 6 --max-plies 40 --stockfish-path local_runtime/stockfish/stockfish/stockfish-windows-x86-64.exe --stockfish-movetime-ms 50 --stockfish-skill-level 1
```

Input data:

- 7 human prompt checks built into `product_demo/evaluate_demo.py`
- 7 prompts loaded from `product_demo/training_data/eval_sft.jsonl`
- Router artifact: `results/production_router_linear/router_model.json`
- Tool backend: `product_demo/chess_tool_demo.py`

Metrics:

| Metric | Result |
|---|---:|
| Test prompts | 14 |
| Tool-name correct | 14/14 = 100.0% |
| Full tool-call correct | 14/14 = 100.0% |
| Tool ran successfully | 14/14 = 100.0% |
| Macro F1 | 100.0% |

Per tool:

| Tool | Correct rate | Prompt count | Support gate |
|---|---:|---:|---|
| `best_move` | 100.0% | 4 | Below 10 prompts |
| `eval` | 100.0% | 4 | Below 10 prompts |
| `review_move` | 100.0% | 6 | Below 10 prompts |

Sample real prompts and responses:

| Prompt | Expected | Predicted | Tool result |
|---|---|---|---|
| `Can you evaluate this position for me?` | `eval` | `eval` | `Current engine bucket: balanced (0 cp from White perspective).` |
| `What move should I consider here?` | `best_move` | `best_move` | `best_move=e2e4`, score `30` |
| `Find one practical candidate move.` | `best_move` | `best_move` | `best_move=e2e4`, score `30` |
| `Was my move e2e4 good?` | `review_move` | `review_move` | `e2e4 is good. Best known alternative: e2e4.` |

Main remaining router caveat:

- Accuracy is perfect on this small local test, but support is only 4/4/6 prompts per tool. This is enough for a proof of concept, not enough for a production claim.
- The passing router artifact is a local linear router, not the Qwen/local-transformer artifact. The Qwen path still needs `transformers` installed and rerun before calling this an LLM router.

### Chess engine — learned linear evaluator against Stockfish

Command used:

```bash
python product_demo/train_chess_engine.py --input product_demo/sample_kaggle_fens.csv --out-dir results/production_chess_stockfish --games 6 --max-plies 40 --stockfish-path local_runtime/stockfish/stockfish/stockfish-windows-x86-64.exe --stockfish-movetime-ms 50 --stockfish-skill-level 1 --use-stockfish-engine --stockfish-engine-movetime-ms 50 --stockfish-engine-skill-level 3
```

Input data:

- Training/eval FEN file: `product_demo/sample_kaggle_fens.csv`
- Training positions: 4
- Eval positions: 1
- Legal move system: `python-chess`
- Stockfish binary: `local_runtime/stockfish/stockfish/stockfish-windows-x86-64.exe`
- Stockfish probe succeeded with `uciok` and `readyok` before match use.

Learned evaluator vs Stockfish level 1:

| Metric | Result |
|---|---:|
| Games | 6 |
| Wins | 0 |
| Losses | 6 |
| Drawish | 0 |
| Score rate | 0.0% |
| Max plies | 40 |

This fails the production chess-engine bar. It is a legal-move proof of concept, not a strong engine.

### Chess engine — Stockfish-backed product mode against weaker Stockfish

Same command also ran a Stockfish-backed engine mode so the product can demonstrate a real engine fight inside the same harness.

| Metric | Result |
|---|---:|
| Product engine | Stockfish, 50 ms/move, skill level 3 |
| Opponent | Stockfish, 50 ms/move, skill level 1 |
| Games | 6 |
| Wins | 3 |
| Losses | 1 |
| Drawish | 2 |
| Score rate | 66.7% |

Sample game 1:

| Field | Value |
|---|---|
| Engine color | White |
| Outcome | Win |
| Plies | 40 |
| Final eval | +1104 cp for White |
| First moves | `e2e3 d7d5 g1f3 a7a6 b2b4 b8c6 c2c4 e7e5` |

This proves Stockfish integration and real local engine-vs-engine play. It does not prove the custom learned evaluator is production strength.

## Files produced

- `results/production_router_linear/router_model.json` — passing local router artifact
- `results/production_router_linear/sft_eval.json` — router training/eval metrics
- `results/production_chess_stockfish/chess_engine_model.json` — chess training, learned-vs-Stockfish, and Stockfish-vs-Stockfish metrics
- `results/production_final_eval/summary.md` — generated final metric summary
- `results/production_final_eval/sft_prompt_simulation.json` — prompt-level router/tool results
- `results/production_final_eval/engine_match_results.json` — final learned-engine match details
- `results/production_final_eval/engine_backend.json` — backend and Stockfish status

## Code changes made

- `product_demo/train_sft_poc.py` now supports the existing `multinomial-naive-bayes-router-v1` artifact.
- `product_demo/train_chess_engine.py` now supports Stockfish UCI opponents and an optional Stockfish-backed engine-vs-Stockfish match.
- `product_demo/evaluate_demo.py` now uses Stockfish for the final match when a working Stockfish path is supplied.
- `.gitignore` excludes local runtime/dependency folders so the downloaded Stockfish binary and large local files stay out of git.

## Final readiness call

| Area | Status | Reason |
|---|---|---|
| Router proof of concept | Pass | 14/14 correct real local tool calls |
| Router production claim | Not yet | Test set too small; passing router is linear, not LLM/Qwen |
| Learned chess evaluator | Fail | Lost 6/6 against Stockfish level 1 |
| Stockfish-backed chess product mode | Pass as integration | Real Stockfish binary ran locally and beat weaker Stockfish 3-1-2 |
| Fake/staged output risk | Pass | Commands produced local JSON/markdown outputs from real code |

Next required work before a true production-grade claim:

1. Install isolated `transformers` dependencies and rerun the Qwen/local-transformer router path, or stop calling the router an LLM.
2. Expand router eval to at least 30 prompts with 10+ per tool.
3. Replace the learned chess evaluator with Stockfish-backed engine mode for product use, or train/evaluate on a real chess corpus large enough to compete.
4. Add hard readiness gates so final pass fails automatically when support counts, LLM-router availability, or Stockfish match strength are below target.
