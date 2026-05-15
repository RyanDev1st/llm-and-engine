# Final Pass Report — Real Local Evaluation

Generated: 2026-05-15

## Bottom line

Not production-grade yet.

The local proof of concept runs real code and real local test files. No staged screenshots, canned success output, or fake Stockfish result was used. It is still below production bar because router accuracy is 71.4%, each tool has fewer than 10 eval examples, and Stockfish is not installed in this environment.

## What was tested

### Tool router

Command used:

```bash
python product_demo/evaluate_demo.py --model product_demo/trained_router.json --eval product_demo/training_data/eval_sft.jsonl --out-dir results/final_pass_eval --games 6 --max-plies 40
```

Input data:

- 7 human prompt checks built into `product_demo/evaluate_demo.py`
- 7 prompts loaded from `product_demo/training_data/eval_sft.jsonl`
- Router artifact: `product_demo/trained_router.json`
- Tool backend: `product_demo/chess_tool_demo.py`

Code fix needed before this could run:

- `product_demo/train_sft_poc.py` now supports the existing `multinomial-naive-bayes-router-v1` router artifact.
- Reason: evaluator previously assumed every non-LLM router had `labels`, `weights`, and `bias`; `trained_router.json` stores `class_counts`, `token_counts`, and `vocabulary`.

Metrics:

| Metric | Result |
|---|---:|
| Test prompts | 14 |
| Tool-name correct | 10/14 = 71.4% |
| Full tool-call correct | 10/14 = 71.4% |
| Tool ran successfully | 11/14 = 78.6% |
| Macro F1 | 71.0% |

Per tool:

| Tool | Correct rate | Prompt count | Production support gate |
|---|---:|---:|---|
| `best_move` | 50.0% | 4 | Fail: below 10 prompts |
| `eval` | 75.0% | 4 | Fail: below 10 prompts |
| `review_move` | 83.3% | 6 | Fail: below 10 prompts |

Sample real prompts and responses:

| Prompt | Expected | Predicted | Tool result |
|---|---|---|---|
| `Can you evaluate this position for me?` | `eval` | `eval` | `Current engine bucket: balanced (0 cp from White perspective).` |
| `What move should I consider here?` | `best_move` | `best_move` | `best_move=e2e4`, score `30` |
| `Find one practical candidate move.` | `best_move` | `review_move` | Error: `Move must be UCI like e2e4 or e7e8q.` |
| `Was my move e2e4 good?` | `review_move` | `review_move` | `e2e4 is good. Best known alternative: e2e4.` |
| `Please review b1c3 without guessing.` | `review_move` | `review_move` | `b1c3 is good. Best known alternative: e2e4.` |

Main failure pattern:

- 3 prompts routed to a tool that needed missing move arguments, so the tool returned an error.
- 1 prompt routed to the wrong tool but still produced an `ok` tool result.

### Chess engine

Same evaluation command also ran the engine match suite.

Input data:

- Engine artifact: `product_demo/poc_models/chess_engine_model.json`
- Legal move system: `python-chess`
- Starting position: standard chess start
- Opponent: seeded capture/random baseline using legal moves
- Games: 6
- Max plies per game: 40

Match metrics:

| Metric | Result |
|---|---:|
| Games | 6 |
| Wins | 4 |
| Losses | 1 |
| Drawish | 1 |
| Score rate | 75.0% |
| Zero-ply games | 0 |
| Plies min/mean/max | 30 / 38.3 / 40 |
| Legal move backend | `python-chess` |

Game results:

| Game | Engine color | Outcome | Plies | End state |
|---:|---|---|---:|---|
| 1 | White | Win | 40 | White better |
| 2 | Black | Loss | 40 | White better |
| 3 | White | Win | 40 | White better |
| 4 | Black | Win | 30 | Checkmate |
| 5 | White | Win | 40 | White better |
| 6 | Black | Drawish | 40 | Balanced |

Separate training/eval run:

Command used earlier:

```bash
python product_demo/train_chess_engine.py --input product_demo/sample_kaggle_fens.csv --out-dir results/final_pass_chess --games 6 --max-plies 20
```

Output:

| Metric | Result |
|---|---:|
| Training positions | 4 |
| Eval positions | 1 |
| Final train accuracy | 100% |
| Eval accuracy | 100% |
| Legal prediction rate | 100% |
| Match score rate | 100% |
| Engine wins | 6 |
| Engine losses | 0 |

This proves the local chess code runs legal moves, but the sample is far too small for production claims.

## Stockfish check

Stockfish result:

```json
{"available": false, "blocker": "stockfish_not_found", "path": null, "uci_ready": false}
```

No Stockfish fight was claimed. The fallback opponent was the seeded legal-move capture/random bot already present in the repo.

## Production readiness

### Passes

- Real local commands executed.
- Router called real chess tools.
- Chess engine used `python-chess` legal move generation.
- Evaluation files were written under `results/final_pass_eval/` and `results/final_pass_chess/`.
- No large files over 5 MB found by audit.
- Secret pattern audit found no obvious API keys, tokens, or passwords in tracked-sized text files.

### Fails / blockers

- Router accuracy is 71.4%, below production target.
- Tool-call success is 78.6%, below production target.
- Per-tool prompt support is too small: 4, 4, and 6 prompts; target gate is 10+ each.
- Stockfish is not installed, so no Stockfish calibration was run.
- Chess training sample is only 5 FEN rows total, with 4 train and 1 eval.
- Current chess opponent is not a rated model and does not prove ELO strength.
- Qwen/local-transformer router artifact still requires missing `transformers`; this final run used the existing Naive Bayes router artifact instead.

## Files produced

- `results/final_pass_eval/summary.md` — generated metric summary
- `results/final_pass_eval/sft_prompt_simulation.json` — prompt-level router/tool results
- `results/final_pass_eval/engine_match_results.json` — engine match details
- `results/final_pass_eval/engine_backend.json` — backend and Stockfish status
- `results/final_pass_chess/chess_engine_model.json` — trained chess evaluator artifact and metrics

## Next required work

1. Install or bundle Stockfish in an isolated runtime and rerun engine matches against it.
2. Expand eval set to at least 30 router prompts, with 10+ per tool.
3. Train/run the local transformer router path or remove it from production claims.
4. Add a hard readiness gate: fail final pass if router/tool accuracy is below target or Stockfish is unavailable.
5. Re-run final pass and only mark production-ready after those gates pass.
