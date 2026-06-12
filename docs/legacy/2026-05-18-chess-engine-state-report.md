Parent: none

## Status

Chess engine is small, deterministic, rule-complete for core chess, and currently reliable under project tests. It is reliable as an LLM tool-call backend and controlled demo engine. It is not yet strong by serious engine standards.

Latest verified state:

- Full engine tests: `73 passed`
- Spec contract check: output `28`
- Demo output stable
- Latest kept commit: `f39916bf experiment: reward connected passed pawns`
- No push or PR done
- Engine commits did not stage or commit `legacy/` changes; current worktree still contains unrelated pre-existing `legacy/` deletions

## Scope

This report covers current chess-engine reliability, precision, scalability, kept experiments, discarded experiments, and recommended next work for advisor handoff.

Hard constraints to preserve:

```text
! Do not touch ./legacy/ ignore it.
```

```text
THe chess engine must be able to receive tool calls from llm per @chess_assistant_sft_dataset_spec_v3.md. This is the ultimate truth for context too.
```

```text
Requirement: DO not ask for spec, the user is asleep. Continuously research and build. The user will review different versions you propose later.
```

```text
Push / PR only when the user explicitly requests it.
```

```text
Secrets: never paste keys, cookies, tokens, or private URLs.
```

## Evidence

### Verification

Latest verified commands:

```text
PYTHONPATH=src python -m pytest tests/engine -q
73 passed
```

```text
PYTHONPATH=src python -m engine.research.spec_check
28
```

```text
PYTHONPATH=src python -m engine.research.demo
```

Stable demo output:

```text
<tool>move san=e4</tool>
success: e4
<tool>best_move depth=15</tool>
best: a5, requested_depth=15, searched_plies=3
<tool>move san=a7a5</tool>
success: a7a5
<tool>legal_moves square=g1</tool>
legal: [Ne2, Nf3, Nh3]
<tool>move san=Nf3</tool>
success: Nf3
<tool>eval depth=15</tool>
score: +0.00 pawns from white POV, requested_depth=15, searched_plies=3
<tool>review_move</tool>
review: Nf3, label=mistake, delta=-1.20 pawns, best_was=Qg4
<tool>list_pieces color=mine</tool>
pieces: P=a5, R=a8, P=b7, N=b8, P=c7, B=c8, P=d7, Q=d8, P=e7, K=e8, P=f7, B=f8, P=g7, N=g8, P=h7, R=h8
```

### Reliability

Current reliability: good for demo and contract backend.

Covered areas:

- Legal move generation
- Self-check filtering
- No king captures
- Castling legality, including through-check filtering
- SAN parsing/formatting
- Promotions
- En passant
- Checkmate/stalemate reporting
- Insufficient-material draws
- Same-colored bishop draw
- Fifty-move draw
- Threefold repetition draw
- Undo support
- Backend tool dispatch

Backend remains compatible with `chess_assistant_sft_dataset_spec_v3.md`.

Canonical API preserved:

- `move`
- `eval`
- `best_move`
- `review_move`
- `threats`
- `legal_moves`
- `undo`
- `list_pieces`
- `ask_chessbot`

Assistant tool-call input format preserved:

```text
<tool>NAME arg=value arg=value</tool>
```

Backend result strings are deterministic but do not exactly match all spec examples: `eval`, `best_move`, and `threats` include diagnostic fields such as `requested_depth` and `searched_plies`. Treat the report's exact-format claim as applying to assistant tool-call inputs, not backend output shapes.

Important behavior preserved:

- Backend owns chess state.
- Backend owns chess logic.
- LLM only routes tool calls and conversational glue.
- Model must not invent results.
- One tool call max per assistant turn.
- Assistant must not call tool after tool result.
- Runtime backend parses one tool-call string but does not enforce full conversation-turn policy; that policy lives in the spec/evaluator layer.

Main reliability risk: tests assert exact backend strings. This is good for regression control but brittle. Small search/eval changes can shift exact output even when chess behavior improves.

Observed fixture drift:

- Search center tie-break changed `threats` best reply from `a5` to `d5`.
- Unsafe queen mobility changed review score from `-9.00` to `-9.15` / `-9.38`.

Conclusion: current reliability strong for fixed fixtures, but tuning must remain tiny and verified across full backend suite.

### Precision

Current chess precision: low-to-moderate.

Search behavior:

- User depth requests like `depth=15` are accepted.
- Actual searched plies clamp to `1..3`.
- Demo confirms `requested_depth=15` and `searched_plies=3`.

This makes output stable and fast but not deeply precise.

Tactical precision currently includes:

- Mate-in-one priority
- Checking move priority
- Capture priority
- Promotion priority
- Castling priority
- Negamax alpha-beta over legal continuations

Missing tactical infrastructure:

- Quiescence search
- Static exchange evaluation
- Transposition table
- Killer moves
- History heuristic
- Null-move pruning
- Late move reductions
- Check extensions
- Iterative deepening
- Time management

Evaluation currently includes:

- Material
- Pawn structure
- Passed pawn reward
- Connected passed pawn bonus
- Doubled/isolated pawn penalty
- Minor-piece center activity
- Rook mobility
- Rook seventh-rank activity
- Bishop mobility
- Knight mobility in simplified positions
- Bishop pair in simplified positions
- Bare-king activity
- Guarded lone-queen mobility in simplified endgames

Recent kept heuristics:

1. `d85b0093 experiment: reward knight mobility`
   - Rewards active knights in simplified positions.

2. `bae9a793 experiment: reward rook activity`
   - Rewards rook on seventh rank.

3. `07760279 experiment: reward active lone queen`
   - Adds queen activity idea.

4. `b57284c2 experiment: guard queen mobility to endgames`
   - Restricts queen mobility to simplified endgames, exactly one queen, queen belongs to side to move, half-weighted mobility.

5. `f39916bf experiment: reward connected passed pawns`
   - Adds +5 when passed pawn has adjacent passed pawn within one rank.

Precision caveat: evaluation is handcrafted and not calibrated against Stockfish or engine-vs-engine benchmarks. Scores are useful internally but not exact centipawn truth.

Spec allows eval tolerance:

```text
numeric values within ±0.30 pawns OR same sign+magnitude class
```

Tests are stricter than spec because exact strings are asserted.

### Scalability

Runtime scalability: limited.

Current search depth is capped at 3 plies. Current engine is responsive and suitable for demo, but not scalable to strong play.

Likely current traits:

- Legal move generation per node
- Alpha-beta pruning
- No persistent transposition table
- No move cache
- No iterative deepening
- No parallel search
- No opening book
- No endgame tablebase

Directly raising depth cap would likely scale poorly due branching factor.

Code scalability: moderate.

Strengths:

- Engine split across focused modules under `src/engine/research/`.
- Tests under `tests/engine/`.
- Tool backend separated from search/eval.
- Evaluation terms are simple functions.
- Experiments easy to add and revert.

Risks:

- `evaluation.py` accumulating many small heuristics.
- Exact backend fixture tests make tuning slow.
- No benchmark harness used in latest loop.
- No performance profiling active.
- No parameter config/tuning mechanism.
- Some heuristic gates are hardcoded to avoid fixture drift.

Product scalability: good for LLM-tool backend.

Strengths:

- Tool-call parser/dispatcher stable.
- Engine state hidden behind backend.
- LLM integration does not require LLM to know chess.
- Tool results are deterministic.
- Demo proves multi-turn flow.

Weaknesses:

- No concurrency/session isolation report in current context.
- No durability/persistence described.
- No load testing.
- No service API scaling analysis.
- Current engine likely single-process, local runtime.

### Quality scorecards

Reliability:

| Area | State | Confidence |
|---|---:|---:|
| Legal move generation | Good | Medium-high |
| Special rules | Good | Medium-high |
| Backend tool contract | Good | High |
| Demo stability | Good | High |
| Eval regression safety | Fair | Medium |
| Search correctness | Basic | Medium |
| Tactical precision | Weak/moderate | Medium |
| Positional precision | Basic/improving | Medium |
| Performance at depth >3 | Unknown/poor likely | Low |
| ELO estimate | Unknown | Low |

Precision:

| Capability | Current |
|---|---|
| Material eval | Present |
| Pawn structure | Present |
| Passed pawns | Present |
| Connected passed pawns | Present |
| Piece activity | Present |
| Rook activity | Present |
| Bishop pair | Present |
| Mobility | Present for bishop/rook/knight/guarded queen |
| King activity | Present in bare-king/endgame form |
| King safety in middlegame | Minimal/unknown |
| Threat detection | Shallow-search based |
| Blunder review | Shallow-search based |
| Mate detection | Present for shallow search |
| Quiescence | Missing |
| Tapered eval | Missing |
| Opening book | Missing |
| Tablebases | Missing |

Scalability:

| Area | Current | Needed |
|---|---|---|
| Search depth | 3 plies max | Iterative deepening + time control |
| Node reuse | None known | Transposition table |
| Capture horizon | Vulnerable | Quiescence |
| Move ordering | Basic priority | TT move + MVV-LVA + killer/history |
| Eval tuning | Manual constants | Test/benchmark-driven tuning |
| Parallelism | None known | Optional later |
| LLM backend API | Stable | Session/load handling later |
| Benchmarking | Tests + demo | Tactical/positional suites |

### Experiments kept

#### `d85b0093 experiment: reward knight mobility`

Added simplified-position knight mobility.

Purpose:

- Improve endgame/minor-piece activity.
- Reward centralized knights with more legal jumps.

Status:

- Kept.
- Verified.
- Backend stable.

#### `bae9a793 experiment: reward rook activity`

Added rook seventh-rank activity.

Purpose:

- Reward classic rook activity in endgames.
- Improve positional play.

Status:

- Kept.
- Verified.
- Backend stable.

#### `07760279 experiment: reward active lone queen`

Initial active queen mobility experiment.

Purpose:

- Improve queen endgame play.
- Reward open queen lines.

Status:

- Kept after later guard commit.

#### `b57284c2 experiment: guard queen mobility to endgames`

Tightened queen mobility.

Purpose:

- Stop queen mobility from drifting tactical backend fixtures.
- Only apply in safe simplified endgames.

Final behavior:

- Ignore crowded positions.
- Ignore queen trades.
- Ignore lone queen when it is not side to move.
- Half-weight mobility score.

Status:

- Kept.
- Verified.
- Backend stable.

#### `f39916bf experiment: reward connected passed pawns`

Added connected passed pawn bonus.

Purpose:

- Reward coordinated passed pawns.
- Small safe eval improvement inside existing pawn structure.

Final behavior:

- If pawn is passed:
  - base passed pawn bonus applies.
  - if adjacent file has own passed pawn within one rank, add +5.

Tests added:

- Connected passed pawns score exact `50`.
- Split passed pawns score exact `0`.
- Start pawn structure score exact `0`.

Status:

- Kept.
- Verified:
  - `73 passed`
  - spec check `28`
  - demo stable

### Experiments discarded

#### Search center-target move-order tie-break

Goal:

- Prefer central target squares in move ordering.

Result:

- Focused search tests passed.
- Full backend fixture drifted.

Failure:

```text
threats: best reply is a5
changed to:
threats: best reply is d5
```

Decision:

- Reverted/discarded.
- Search ordering affects exact backend behavior too much.

#### Unsafe queen mobility variants

Goal:

- Reward queen mobility broadly.

Failures:

```text
review: Qf2, label=blunder, delta=-9.00 pawns
changed to:
review: Qf2, label=blunder, delta=-9.15 pawns
```

and:

```text
-9.00
changed to:
-9.38
```

Decision:

- Reworked into guarded endgame-only queen mobility.
- Final guarded version kept.

### Current strength estimate

No measured ELO. Do not claim ELO.

Likely strength:

- Above random legal move generator.
- Handles rules correctly.
- Sees shallow tactics.
- Has basic positional heuristics.
- Weak versus real engines due 3-ply cap and no quiescence.

Best characterization:

```text
Reliable demo engine, not competitive engine.
```

Biggest strength blockers:

1. Quiescence search
2. Transposition table
3. Iterative deepening + time control
4. Better move ordering
5. Evaluation phase awareness
6. Benchmark positions

### Independent verification notes

Three read-only subagents verified this report after drafting:

- Scalability/precision verifier: PASS for depth clamp `1..3`, missing quiescence/transposition table/iterative deepening/time control/opening book/tablebase/parallel search, and negamax alpha-beta plus static-eval architecture.
- Contract verifier: PASS for canonical 9 tools, backend-owned state/logic, spec-level one-tool-call constraints, stable demo snapshot, and brittle exact backend tests. It flagged that exact tool format applies to assistant inputs, while backend outputs include extra diagnostics.
- Code/test verifier: PASS for latest commit `f39916bf`, connected passed pawn bonus, knight/rook/queen heuristic wiring, `73 passed`, and spec check `28`. It flagged that current worktree contains unrelated `legacy/` deletions, though engine commits did not stage or commit them.

## Next

1. Add benchmark harness before deeper changes:
   - tactical positions
   - endgame positions
   - backend fixture snapshot
   - runtime timing

2. Consider quiescence search as highest-value precision upgrade:
   - captures only first
   - checks later
   - hard node cap
   - expect possible fixture drift

3. If fixture drift occurs, advisor should decide whether exact tests should loosen to spec tolerance.

4. Add transposition table after quiescence decision.

5. Safer near-term alternative: add tightly gated endgame eval improvement, such as:
   - rook behind passed pawn
   - king supports passed pawn
   - pawn advancement in king-pawn endgames
   - opposition/simple king distance

6. Keep commits as `experiment: ...`, discard failed experiments, and verify before claiming done.
