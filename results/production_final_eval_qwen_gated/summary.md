# SFT Router and Chess Engine Evaluation

Generated: 2026-05-15T04:25:51.035550+00:00

## SFT Router Prompt Evaluation

- Cases: 37
- Router tool-name accuracy: 1.000 (37/37)
- Router end-to-end tool-call accuracy: 1.000 (37/37)
- Tool success rate: 1.000 (37/37)
- Metrics by board source:
  - default_start: tool_accuracy=1.000 (7/7), end_to_end_accuracy=1.000 (7/7), tool_success_rate=1.000 (7/7)
  - eval_fen: tool_accuracy=1.000 (30/30), end_to_end_accuracy=1.000 (30/30), tool_success_rate=1.000 (30/30)
- Where it succeeds:
  - best_move: 1.000 over 12 prompts, minimum_support_passed=True
  - eval: 1.000 over 12 prompts, minimum_support_passed=True
  - review_move: 1.000 over 13 prompts, minimum_support_passed=True
- Why failures happen:
  - No prompt evaluation failures recorded.

## Engine Match Evaluation

- Learned evaluator model: basic-linear-python-chess-evaluator-v1
- Legality backend: python-chess
- Learned evaluator opponent: Stockfish UCI opponent at 50ms/move, skill_level=1
- Learned evaluator games: 6
- Learned evaluator score rate: 0.000
- Learned evaluator zero-ply games: 0

## Stockfish-Backed Product Engine

- Product engine: Stockfish UCI engine at 50ms/move, skill_level=4
- Opponent: Stockfish UCI opponent at 50ms/move, skill_level=1
- Games: 6
- Wins/losses/drawish: 5/1/0
- Score rate: 0.833

## Final Readiness Gate

- Passed: True
  - No blockers recorded.

## Proficiency Conclusion

Product path uses a local Qwen transformer router for tool calls and a Stockfish-backed chess engine for play. The learned linear evaluator is retained only as an experimental baseline, not as the product chess engine.
