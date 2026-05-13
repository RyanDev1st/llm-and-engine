# Correctness-first pipeline spec: chess-engine-backed LLM tool calling

**Date:** 2026-05-13
**Status:** adversarially revised draft
**Priority:** correctness first
**Scope:** end-to-end pipeline from research evidence to release gates

## Executive decision

The chess assistant should not be built dataset-first. The first build artifact should be a versioned executable task environment: policy, local tools, deterministic board/session state, task scenarios, trace schema, replay runner, and evaluator. Dataset examples are accepted only after they replay successfully and pass policy-grounded grading.

Core principle:

> Environment defines truth. Runtime enforces channels. Evaluator grades trajectories. Dataset is only replayable evidence. Model learns routing and grounded narration, never chess authority.

## Why the prior v3 spec must change

The v3 spec has useful ingredients: FEN-blind state, local python-chess/Stockfish backend, tool replay, strict tool grammar, and model-as-router rather than model-as-chess-authority. But it is too weak in four places:

1. It starts from SFT data generation instead of an executable environment and evaluator.
2. It relies on a unified prompt for Mode 1 / Mode 2 discipline instead of API-level channel isolation.
3. It includes `ask_chessbot` as a broad fallback, which can contaminate correctness evaluation.
4. It uses approximate target counts and validation rates before baseline distributions identify real failure buckets.

The corrected pipeline keeps the backend-oracle idea but makes environment, policy, and replay the center of the system.

## Non-negotiable invariants

### Chess truth

- The model never owns board truth.
- The model never computes legal moves, evaluations, tactics, mate status, material, threats, or move quality.
- Any position-specific claim must be grounded in a validated tool result.
- Backend state is authoritative and replayable.

### Runtime isolation

- Router may emit a strict structured tool call or a direct-answer class.
- Narrator cannot call tools by API design.
- Runtime never executes tool-like text produced by narrator.
- Narrator receives only validated tool results and policy-safe context, not raw backend logs.

### Dataset discipline

- No example enters training unless it replays against the environment.
- No train/dev/test split by random example alone.
- Splits must avoid leakage by exact FEN, game source, position family, motif, opening family, scenario template, and paraphrase cluster.
- SFT data is evidence produced by environment and evaluator, not hand-authored target behavior.

### Evaluation discipline

- Correctness scoring is programmatic where possible.
- LLM judges may grade coaching style only after correctness gates pass.
- Release requires zero blocker failures.
- Quality thresholds are set from baseline distributions, not guessed up front.
- Every gate needs an adversarial counterexample suite: wrong trace, wrong route, wrong state, wrong claim, and prompt-injection variant.

## Phase 0 — Search-first prior art ledger

**Goal:** Maintain evidence trail before architecture hardens.

**Artifact:** `literature/NOTES.md` plus source-linked findings.

**Required evidence classes:**

- chess backend docs: python-chess, Stockfish/UCI behavior
- inference backend docs: vLLM structured outputs, llama.cpp grammars, OpenAI-compatible strict schemas
- tool-calling benchmark repos: tau-bench/tau2, ToolBench, StableToolBench, Gorilla/BFCL, VAKRA, tool-eval-bench
- chess-specific implementation search, including negative results if no canonical chess LLM tool-calling pipeline exists

**Gate:** Every design claim is either source-backed or labeled assumption.

## Phase 1 — Domain contract bundle

**Goal:** Freeze the environment contract before data generation.

**Artifact:** versioned domain contract bundle.

**Bundle contents:**

1. tool schemas
2. result schemas
3. error schemas
4. board/session state model
5. policy text
6. scenario schema
7. evaluator rules
8. trace envelope
9. version manifest

**Contract rules:**

- Use JSON Schema or equivalent strict schemas.
- Unknown fields rejected.
- Required fields explicit.
- State-mutating tools require idempotency key.
- Tool results use structured objects internally, even if narrator sees a compact rendering.
- All schemas carry version IDs.
- Separate policy, scenario expectation labels, and evaluator implementation so grading cannot become circular.
- Define visibility levels: internal replay metadata, model-visible context, and user-visible answer.

**Minimum schema skeletons:**

- tool call envelope: `schema_version`, `tool_name`, `arguments`, `session_id`, `idempotency_key`, `visibility`, `requested_at`.
- tool result envelope: `schema_version`, `tool_name`, `tool_call_id`, `status`, `state_delta`, `evidence`, `engine_profile_id`, `visibility`, `completed_at`.
- error envelope: `schema_version`, `tool_name`, `tool_call_id`, `error_code`, `retryable`, `safe_user_message`, `internal_detail_ref`.
- session state snapshot: `schema_version`, `session_id`, `position_hash`, `fen_internal`, `move_stack`, `side_to_move`, `castling_rights`, `ep_square`, `halfmove_clock`, `fullmove_number`, `engine_profile_id`.
- scenario schema: `scenario_id`, `pack_version`, `initializer`, `user_task`, `expected_route`, `allowed_tool_sequence`, `expected_state_transition`, `grounding_requirements`, `split_metadata`.
- trace envelope: `trace_id`, `scenario_id`, `environment_version`, `policy_version`, `tool_schema_version`, `events`, `final_answer`, `evaluation_result`, `replay_hash`.
- version manifest: `environment_version`, `policy_version`, `tool_schema_version`, `scenario_pack_versions`, `evaluator_version`, `engine_profile_id`, `created_at`.

**Initial MVP tool set:**

Keep position tools:

- `move`
- `eval`
- `best_move`
- `review_move`
- `threats`
- `legal_moves`
- `undo`
- `list_pieces`

Remove from core correctness MVP:

- `ask_chessbot`

Optional later non-position education path:

- `explain_concept`, isolated from stateful board tools and excluded from position-correctness scoring.

**Gate:** Schema validation passes; bad traces with unknown tools, unknown fields, missing required fields, bad types, and invalid state mutation are rejected.

## Phase 2 — Local executable chess environment

**Goal:** Build deterministic environment before any model training.

**Artifact:** environment runner.

**Components:**

- python-chess board/session core
- Stockfish/UCI adapter
- deterministic engine settings
- scenario initializer
- tool executor
- trace recorder
- replay runner
- evaluator hooks

**State model requirements:**

- session ID
- board state held by backend, not model
- move stack
- side to move
- tool-call IDs / idempotency keys
- engine settings version
- scenario ID and environment version
- FEN/hash stored only as internal replay metadata, never model-visible unless product policy changes

**Engine reproducibility profile:**

- exact Stockfish binary hash and version
- exact NNUE/EvalFile hash
- `Threads=1`
- fixed `Hash`
- explicit `Clear Hash` before each independent scenario, or documented persistent-hash policy
- fixed `MultiPV`
- fixed `depth` or `nodes` for correctness packs; avoid wall-clock `time` for release-critical grading
- trace records depth, seldepth, nodes, elapsed time, score, PV, and all engine options
- evaluator separates exact deterministic packs from normalized tolerant packs

**Tool behavior requirements:**

- `move`: execute only explicit SAN/UCI-like moves after ambiguity is resolved; mutate state only on success.
- ambiguous natural-language move requests route to resolver/clarification behavior before `move`.
- `undo`: mutate state only if move stack non-empty.
- `eval`: read-only engine analysis.
- `best_move`: read-only engine PV extraction.
- `review_move`: analyze copied predecessor/current positions and never mutate live session state.
- `threats`: read-only opponent-intent approximation with documented semantics.
- `legal_moves`: read-only legal move list.
- `list_pieces`: read-only piece inventory.

**Gate:** Every seed scenario can run without an LLM and produce expected oracle traces.

## Phase 3 — Policy-first router/narrator runtime

**Goal:** Make prompt failures non-executable.

**Artifact:** baseline runtime harness.

**Router:**

- Has tool-call channel.
- Emits either strict tool call or direct-answer class.
- Cannot narrate unvalidated tool results.
- Must choose no-tool class for greetings, meta, off-topic, and non-position education.

**Backend:**

- Validates tool schema.
- Executes only known tools.
- Applies timeout and deterministic error handling.
- Records normalized trace.

**Narrator:**

- Has no tool-call channel.
- Receives validated tool result plus policy context.
- Produces user-facing answer.
- Cannot cause tool execution even if it emits tool-like text.

**Gate:** Mode 2 executable tool calls are impossible by API/channel design.

## Phase 4 — Evaluator before dataset

**Goal:** Grade trajectories before creating training examples.

**Artifact:** evaluator CLI/report.

**Evaluator checks:**

- schema validity
- route correctness
- policy adherence
- state transition correctness
- tool result normalization
- final answer groundedness
- final-answer claim taxonomy: board-state, legality, evaluation, tactic/threat, move-quality, generic principle, social/encouragement
- trajectory success
- safety/adversarial behavior
- replay determinism
- evaluator conformance against one-bug-per-trace mutants

**Blocker gates, must be 100%:**

- no schema-invalid executable calls
- no unknown tool names
- no unknown fields
- no invalid state mutation
- no invented tool result in final answer
- no executable narration-phase call
- all state transitions replay
- no position-specific answer without required oracle result

**Evaluator mutant table:**

| Mutant | Injected fault | Required evaluator failure |
|--------|----------------|----------------------------|
| unknown-tool | Replace allowed tool with nonexistent tool | unknown tool blocker |
| missing-arg | Remove required tool argument | schema-invalid blocker |
| wrong-arg | Change legal target square, move, or depth parameter | route/argument mismatch |
| illegal-mutation | Apply state delta after failed `move` | invalid state mutation blocker |
| read-only-mutation | Let `eval`, `best_move`, `review_move`, `threats`, `legal_moves`, or `list_pieces` alter state | invalid state mutation blocker |
| invented-result | Final answer cites tool evidence absent from trace | invented result blocker |
| ungrounded-claim | Final answer makes board/eval/tactic claim without matching evidence | ungrounded final claim blocker |
| fen-leak | Model-visible or user-visible content includes raw FEN unexpectedly | leakage blocker |
| prompt-injection | User/tool/KB text instructs model to ignore policy and model complies | prompt-injection blocker |
| contamination | Dev/test scenario shares exact FEN, source game, template, or paraphrase cluster with train | split contamination blocker |

**Quality gates, measured by bucket:**

- route accuracy
- trajectory success
- final answer helpfulness after correctness
- latency
- cost
- refusal/clarification quality
- recovery from timeout or invalid input

**Gate:** Evaluator grades handcrafted golden traces and injected bad traces correctly before data generation starts. It must also fail one-bug-per-trace mutants covering wrong tool, wrong args, wrong state mutation, invented result, ungrounded final claim, prompt-injection compliance, and leakage violation.

## Phase 5 — Scenario packs

**Goal:** Build versioned task packs with oracle-checkable outcomes.

**Artifact:** scenario pack releases.

**Required packs:**

- legal move
- illegal move
- ambiguous move
- underspecified natural-language move
- best move
- eval
- review move
- threats
- undo/state
- special moves: castling rights, en passant, promotion, check/checkmate/stalemate/draw
- timeout/error
- adversarial/prompt injection
- educational no-tool
- long trajectory
- long-game state drift

**Seed coverage matrix:**

| Pack | Minimum seed cases | Must cover |
|------|--------------------|------------|
| legal move | explicit SAN, explicit UCI, check-giving move | successful mutation only on legal move |
| illegal move | illegal destination, wrong side, move after game over | no mutation and safe error |
| ambiguous move | two knights can capture, two rooks can move, unclear “take it” | clarification or resolver before `move` |
| underspecified natural-language move | “castle,” “recapture,” “push the pawn,” “promote” | resolver/clarification, not guessed SAN |
| eval | quiet position, tactical position, mate score | fixed engine profile plus normalized score class |
| best move | single PV, MultiPV, mate-in-N | PV evidence and no exact-score overclaim |
| review move | good move, blunder, illegal proposed move | copied predecessor/current positions, no live mutation |
| threats | immediate capture, mate threat, no threat | documented threat semantics |
| undo/state | undo normal move, undo after capture, undo at empty stack | correct stack and state restoration |
| special moves | castling rights loss, en passant expiry, promotion choice, checkmate, stalemate, draw | special-rule state drift |
| timeout/error | engine timeout, backend unavailable, malformed tool args | recovery answer without invented result |
| adversarial/prompt injection | user text, tool error field, KB content, move label injection | user-controlled strings treated as data |
| educational no-tool | opening principle, tactic definition, rules question | no current-board claim |
| long-game state drift | 20+ ply fragment with eval/review/undo checkpoints | replay hash and state equality at each checkpoint |

**Scenario schema fields:**

- scenario ID
- pack version
- initial board initializer
- user task text or template
- expected route class
- expected tool sequence constraints
- expected state transitions
- expected final-answer grounding requirements
- split metadata: source game, FEN hash, motif, opening family, template cluster, paraphrase cluster

**Generation policy:**

- Use templated scenario generation first.
- Use LLM paraphrase only after template compiles.
- Use adversarial perturbations only against already-valid scenarios.
- Hold out entire template clusters and paraphrase clusters to measure extrapolation, not memorized pattern matching.
- Human/audit approval required for expected route changes.

**Gate:** Scenario enters eval only after oracle compile and expected route validation.

## Phase 6 — Baseline model evaluation

**Goal:** Measure failures before training.

**Artifact:** baseline report.

**Baselines:**

- prompt-only unified model
- grammar/schema-constrained router
- router/narrator split
- local candidate models in target size range

**Baseline model matrix:**

| Runtime/model class | Purpose | Required report fields |
|---------------------|---------|------------------------|
| prompt-only unified | preserve v3-style baseline | format failures, Mode 2 violations, hallucinated board claims |
| grammar/schema router | isolate format adherence from semantic routing | schema pass rate, wrong-tool rate, wrong-arg rate |
| router/narrator split | test channel isolation and grounded narration | route accuracy, trajectory success, narrator groundedness |
| local 3B candidate | deployment lower bound | latency, cost, route accuracy, blocker failures |
| local 7B/8B candidate | likely quality target | latency, cost, route accuracy, bucket regressions |
| strongest hosted reference | ceiling estimate only, not release dependency | failure taxonomy comparison and oracle disagreement review |

**Measurement protocol:**

- Run identical frozen scenario packs across every baseline.
- Use same engine profile, policy version, schemas, and evaluator version.
- Report per-bucket confidence intervals using bootstrap over scenario IDs.
- Separate format validity, route correctness, state replay, final-answer groundedness, and helpfulness.
- Preserve all failed traces for cluster review before setting training targets.
- Never lower blocker requirements based on baseline weakness.

**Report buckets:**

- format failures
- wrong tool
- right tool wrong args
- illegal state mutation attempt
- missing clarification
- hallucinated board claim
- failed timeout recovery
- narrator ungroundedness
- over-tooling abstract chess questions
- under-tooling position questions

**Gate:** Training target chosen from measured failure distribution.

## Phase 7 — Trace and data generation

**Goal:** Generate replayable training candidates only from validated scenarios.

**Artifact:** SFT candidate pool with replay manifest.

**Data classes:**

- router SFT: user/context to route/tool args/direct class
- narrator SFT: validated tool result to grounded final answer
- hard negatives: tempting wrong route, Mode 2 fake tool text, prompt-injection attempts
- no-tool examples: abstract chess education, off-topic chess words, greetings/meta
- recovery examples: timeout, engine unavailable, invalid syntax, ambiguity

**Acceptance criteria:**

- Candidate replays cleanly.
- Assistant claims are grounded in tool output.
- Split metadata shows no leakage.
- Tool args are schema-valid.
- No raw FEN visible to model if product policy says FEN-blind applies to model-visible context.

**Gate:** Candidate promoted to training only after replay and evaluator pass.

## Phase 8 — Training ladder

**Goal:** Train narrowly and verify after each step.

**Order:**

1. Router SFT for route/tool args/direct-answer class.
2. Narrator SFT for concise grounded explanation.
3. Hard-negative SFT for no-tool, ambiguity, and adversarial cases.
4. Preference/rejection sampling only after correctness gates pass.
5. RL-style simulator training only after deterministic environment and evaluator mature.

**Acceptance rule:**

A checkpoint is accepted only if blocker gates remain perfect and quality gates improve in targeted buckets without regressions in held-out buckets.

**Gate:** No checkpoint ships because it looks good in demos; it must pass replayed held-out eval.

## Phase 9 — Release harness

**Goal:** Release only reproducible, versioned packs.

**Artifact:** release report.

**Release manifest:**

Required fields:

- model ID/checkpoint hash
- environment version
- policy version
- tool schema version
- result/error schema versions
- scenario pack versions and split hashes
- evaluator version and mutant-suite version
- engine binary/version/settings and engine profile hash
- baseline comparison with confidence intervals
- blocker gate table
- quality bucket table
- contamination/leakage report
- known limitations
- rollback plan

**Release gate:**

Pass/fail semantics:

- Missing required manifest field: fail.
- Any blocker gate below 100%: fail.
- Any trajectory replay mismatch: fail.
- Any evaluator mutant not caught: fail.
- Any train/dev/test contamination violation: fail.
- Any model-visible or user-visible FEN leakage outside policy: fail.
- Any regression beyond predeclared threshold against previous accepted release: fail.
- Any quality bucket below calibrated floor: no release unless explicitly re-scoped out of supported product behavior.
- LLM judge helpfulness may only add advisory notes after all correctness gates pass.

## Revised product/tool decisions

### `ask_chessbot`

Remove from core MVP. It blurs correctness boundaries because hard board-position questions can be routed to a broad explanatory fallback. If educational content is needed, use an isolated no-position path such as `explain_concept` and prohibit it from answering current-board claims.

### Unified prompt vs router/narrator

Do not rely on a unified prompt as the runtime safety boundary. It can remain a baseline condition for comparison, but production should use channel isolation: router can call tools, narrator cannot.

### FEN-blind constraint

User-visible and model-visible contexts should remain FEN-blind by default. Internal traces may store FEN/hash/state for replay, leakage detection, and reproducibility, but release policy must explicitly define what is model-visible.

### Engine scores

Do not compare exact engine output unless engine settings are fixed. Normalize centipawn and PV outputs into bins/classes for model-facing narration and evaluator comparisons, while preserving raw engine output in internal trace metadata. Use exact PV/score checks only in deterministic packs with frozen engine profile.

### Prompt injection

Local chess tools reduce external API instability, but user text, KB text, and error fields can still carry instructions. Treat all user-controlled strings in tool results as data: structured, escaped, and never interpreted as runtime instructions.

## Minimum viable release sequence

1. Source ledger and assumptions list.
2. Contract bundle v0.
3. Local environment runner v0.
4. Evaluator with golden and bad traces.
5. Seed scenario pack v0.
6. Baseline report.
7. Replayable SFT candidate pool.
8. Router/narrator training.
9. Held-out eval.
10. Release pack.

## Open assumptions requiring follow-up research

1. Best Stockfish settings for stable cross-machine evaluation.
2. Best normalization for centipawn, mate, and PV outputs without overfitting exact depth noise.
3. Whether target local models route better with native tool-call fine-tuning or JSON grammar plus router prompt.
4. Best style/helpfulness evaluation once correctness passes.
5. Exact boundary for FEN-blind policy in internal traces versus model-visible context.

## Spec review checklist

- Does every position-specific claim require an oracle-backed result?
- Can narrator ever trigger executable tools?
- Can every scenario replay without an LLM?
- Does evaluator exist before dataset generation?
- Are blocker gates binary and 100%?
- Are quality thresholds baseline-derived?
- Are train/dev/test splits leakage-resistant?
- Is every release artifact versioned?
