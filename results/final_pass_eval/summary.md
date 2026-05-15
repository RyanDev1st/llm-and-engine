# SFT Router and Chess Engine Evaluation

Generated: 2026-05-15T03:16:35.578042+00:00

## SFT Router Prompt Simulation

- Cases: 14
- Router tool-name accuracy: 0.714 (10/14)
- Router end-to-end tool-call accuracy: 0.714 (10/14)
- Tool success rate: 0.786 (11/14)
- Metrics by board source:
  - default_start: tool_accuracy=0.857 (6/7), end_to_end_accuracy=0.857 (6/7), tool_success_rate=0.857 (6/7)
  - eval_fen: tool_accuracy=0.571 (4/7), end_to_end_accuracy=0.571 (4/7), tool_success_rate=0.714 (5/7)
- Where it succeeds:
  - best_move: 0.500 over 4 prompts, minimum_support_passed=False
  - eval: 0.750 over 4 prompts, minimum_support_passed=False
  - review_move: 0.833 over 6 prompts, minimum_support_passed=False
- Why failures happen:
  - router=False;tool_status=error: 3
  - router=False;tool_status=ok: 1

## Engine Match Evaluation

- Engine model: basic-linear-python-chess-evaluator-v1
- Legality backend: python-chess
- Starting position: chess.STARTING_FEN
- Stockfish available: False
- Opponent: seeded capture/random baseline using python-chess legal move generation; separate from training-time same-heuristic baseline
- Games: 6
- Plies min/mean/max: 30/38.3/40
- Zero-ply games: 0
- Engine wins: 4
- Engine losses: 1
- Drawish: 1
- Engine score rate: 0.750

## Proficiency Conclusion

Current engine match uses trained basic linear evaluator artifact with python-chess legal move generation and terminal rules. Match opponent is seeded capture/random baseline for lightweight sanity only; training artifacts may report separate same-heuristic baseline metrics. Metrics here are measured locally and are not calibrated ELO; Stockfish availability is reported only as environment smoke check.
