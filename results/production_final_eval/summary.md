# SFT Router and Chess Engine Evaluation

Generated: 2026-05-15T03:37:33.416290+00:00

## SFT Router Prompt Evaluation

- Cases: 14
- Router tool-name accuracy: 1.000 (14/14)
- Router end-to-end tool-call accuracy: 1.000 (14/14)
- Tool success rate: 1.000 (14/14)
- Metrics by board source:
  - default_start: tool_accuracy=1.000 (7/7), end_to_end_accuracy=1.000 (7/7), tool_success_rate=1.000 (7/7)
  - eval_fen: tool_accuracy=1.000 (7/7), end_to_end_accuracy=1.000 (7/7), tool_success_rate=1.000 (7/7)
- Where it succeeds:
  - best_move: 1.000 over 4 prompts, minimum_support_passed=False
  - eval: 1.000 over 4 prompts, minimum_support_passed=False
  - review_move: 1.000 over 6 prompts, minimum_support_passed=False
- Why failures happen:
  - No prompt evaluation failures recorded.

## Engine Match Evaluation

- Engine model: basic-linear-python-chess-evaluator-v1
- Legality backend: python-chess
- Starting position: chess.STARTING_FEN
- Stockfish available: True
- Opponent: Stockfish UCI opponent at 50ms/move, skill_level=1
- Games: 6
- Plies min/mean/max: 30/36.0/40
- Zero-ply games: 0
- Engine wins: 0
- Engine losses: 6
- Drawish: 0
- Engine score rate: 0.000

## Proficiency Conclusion

Current engine match uses trained basic linear evaluator artifact with python-chess legal move generation and terminal rules. Metrics here are measured locally and are not calibrated ELO.
