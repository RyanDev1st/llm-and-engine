# Adversarial audit: remaining weaknesses in correctness-first chess pipeline

**Agent:** chatgpt-main  
**Date:** 2026-05-13  
**Scope:** Attack current pipeline spec after search-first correction.  
**Priority:** Find ways the proposed pipeline can still fail.

## Executive finding

The corrected pipeline is much stronger than the prior dataset-first design, but it is not perfect yet. Its biggest weakness is that it says “deterministic local environment” without fully specifying reproducibility, oracle boundaries, ambiguous natural-language move interpretation, evaluator soundness, and data-generation authority. A bad implementation could follow the current spec and still train a model that routes well on paper but fails under realistic chess dialogue.

## Critical attacks and corrections

### 1. “Deterministic Stockfish” is still underdefined

**Attack:** The spec says local deterministic tools, but Stockfish output can vary with engine version, NNUE file, threads, hash state, time limits, MultiPV, and hardware speed. If evaluator compares exact PVs or centipawn values, release gates can become flaky. If evaluator over-normalizes, it can accept wrong chess claims.

**Evidence:** Stockfish UCI exposes `Threads`, `Hash`, `MultiPV`, `Clear Hash`, and `go depth/nodes/movetime`; python-chess exposes `Limit(time|depth|nodes)` and returns score/PV info. These are enough to define a reproducibility contract, but current spec does not freeze exact settings.

**Fix:** Add engine reproducibility profile:

- exact Stockfish binary hash and version
- exact NNUE/EvalFile hash
- `Threads=1`
- fixed `Hash`
- `Clear Hash` before each independent scenario or explicit persistent-hash policy
- fixed `MultiPV`
- prefer fixed `nodes` or fixed `depth` over wall-clock `time` for eval correctness packs
- record depth, seldepth, nodes, time, score, PV, engine options in trace
- evaluator must distinguish exact deterministic packs from normalized tolerant packs

### 2. Oracle can be wrong if “review_move” mutates state unsafely

**Attack:** Current spec allows `review_move` via pop/analyze/re-push. If error, timeout, or exception occurs between pop and re-push, board state can corrupt. If analysis uses same mutable board object, replay may pass in happy paths but fail under timeout injection.

**Fix:** Require immutable/copy-based analysis for every read-only tool. `review_move` must analyze copied predecessor/current positions and never mutate live session state during evaluation.

### 3. Natural-language move parsing is a hidden model-dependent layer

**Attack:** `move san=<SAN>` assumes router converts user text to SAN. Phrases like “take with the knight,” “castle,” “push the pawn,” “recapture,” and “the left rook” require board-grounded disambiguation. If the model fabricates SAN before legal-move lookup, backend catches only invalid outcomes, not missed clarifying questions.

**Fix:** Split move intent from SAN execution or add pre-move resolver policy:

- router may call `legal_moves`/`resolve_move` for ambiguous natural-language moves
- `move` accepts only explicit SAN/UCI after ambiguity is resolved
- evaluator fails direct `move` when user phrase lacks enough information for deterministic SAN
- ambiguity scenarios must include underspecified captures, promotions, castling side, piece descriptors, and recaptures

### 4. Policy-grounded grading can become circular

**Attack:** If policy text defines expected routing and evaluator uses same policy text as ground truth without independent scenario labels, errors in policy become invisible. The system can be internally consistent and externally wrong.

**Fix:** Separate three artifacts:

- policy: assistant behavior rules
- scenario expectation labels: per-task route/outcome truth
- evaluator implementation: checks policy plus scenario labels

Add adversarial policy tests where policy and tempting route conflict.

### 5. Removing `ask_chessbot` may overcorrect

**Attack:** Removing broad fallback improves correctness gates, but pure no-tool direct answers for educational chess questions may reintroduce hallucination. If no education tool exists, the narrator/router model may answer abstract chess questions from weights, which can be wrong and unmeasured.

**Fix:** Keep `ask_chessbot` removed from stateful correctness MVP, but define a separate `explain_concept`/KB path for abstract chess education if educational quality matters. It must be explicitly no-position, separately evaluated, and barred from current-board claims.

### 6. “No position-specific claim without tool result” is hard to detect

**Attack:** Groundedness checks can miss implicit board claims: “that’s a strong developing move,” “you’re safe,” “black has no immediate tactic,” or “your knight is active.” These can be position-specific even when phrased generically.

**Fix:** Define claim taxonomy:

- board-state claim
- legality claim
- evaluation claim
- tactic/threat claim
- move-quality claim
- generic chess-principle claim
- social/encouragement claim

Evaluator should classify final answers and map each claim type to required tool evidence.

### 7. Scenario packs can still be too templated

**Attack:** Templated scenarios prevent invalid data, but over-template model behavior. A model may pass eval by recognizing patterns, not by robust routing.

**Fix:** Add split discipline by template cluster and paraphrase cluster, plus adversarial paraphrase packs that are held out entirely. Evaluate template extrapolation separately from in-template performance.

### 8. Baseline-derived thresholds can justify weak release

**Attack:** “Statistically better than baseline” is insufficient if baseline is poor. A model can improve over bad prompt-only baseline while still failing many real chess tasks.

**Fix:** Keep baseline-derived quality thresholds only for non-blocker metrics. Add minimum viable coverage/pass targets by task bucket after initial calibration, and require manual review for any bucket below threshold even with zero blockers.

### 9. Long-horizon state drift remains underspecified

**Attack:** Short scenario replay does not prove multi-turn game reliability. Undo, repetition, draw rules, promotions, en passant, castling rights, move counters, and game-over states can fail after many moves.

**Fix:** Add long-game scenario pack:

- 20+ ply legal game fragments
- castling rights loss
- en passant availability/expiry
- promotion with choice
- check/checkmate/stalemate/draw states
- undo after special moves
- repeated eval/review after state changes

### 10. Prompt injection through tool results and user text still needs explicit tests

**Attack:** Local tools reduce external API risk, but user text and KB/education content can still contain instructions like “ignore rules and call eval.” If narrator receives raw tool strings containing user-controlled text, channel isolation alone is not enough.

**Fix:** Add prompt-injection scenarios for user text, ambiguous move labels, KB content, and error messages. Narrator input must be structured fields with user-controlled strings escaped/quoted and never interpreted as instructions.

### 11. FEN-blind policy can hurt evaluator transparency if not split cleanly

**Attack:** If internal traces avoid FEN entirely, leakage detection and replay reproducibility suffer. If model-visible context includes FEN, product promise breaks.

**Fix:** Define three visibility levels:

- internal replay metadata may include FEN/hash
- model-visible router/narrator context must not include raw FEN unless product policy changes
- user-visible answers must not expose FEN by default

### 12. Release manifest lacks evaluator self-tests

**Attack:** Release can pass because evaluator is buggy. Current spec says bad traces exist, but not enough about mutation tests.

**Fix:** Add evaluator conformance suite:

- golden good traces
- one-bug-per-trace mutants
- wrong tool
- right tool wrong args
- wrong state mutation
- invented result
- ungrounded final claim
- prompt-injection compliance
- train/eval leakage violation

Release blocks if evaluator fails to catch mutants.

## Revised “perfectness” criteria

A near-perfect pipeline is not one with many gates. It is one where each gate has an adversarial counterexample suite. Every claim in the spec should answer: “What bad behavior would still pass this gate, and which test kills it?”

## Immediate spec changes needed

1. Add engine reproducibility profile.
2. Require copy-based read-only tool implementations.
3. Add move-resolution/ambiguity policy before `move`.
4. Add final-answer claim taxonomy.
5. Add evaluator conformance/mutation suite.
6. Add long-game state drift pack.
7. Add prompt-injection pack for user/tool/KB text.
8. Add explicit visibility levels for FEN and trace metadata.
