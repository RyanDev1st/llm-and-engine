# Search-first audit: correctness-first chess tool-calling pipeline

**Agent:** chatgpt-main  
**Date:** 2026-05-13  
**Scope:** Audit prior end-to-end pipeline for a chess-engine-backed LLM tool-calling assistant.  
**Priority:** Correctness first; search-first evidence; no Claude subagents.

## Executive finding

Prior pipeline direction was broadly right—contracts, oracle, replay, eval before training—but not strict enough. Search-first audit shows the pipeline must be revised from a generic “dataset/eval/training ladder” into a benchmark-style executable environment: frozen policies, local tools, deterministic state, trajectory replay, policy-grounded grading, and versioned release packs.

Key correction: the first build artifact is not the dataset and not even the tool schema alone. It is the **executable task environment**: policy + tools + state database + user/task scenarios + evaluator. This follows the strongest pattern from τ-bench, VAKRA, StableToolBench, and deterministic local tool-calling benchmarks.

## Search-first evidence

### GitHub / repo search

- Direct GitHub search for chess-specific LLM+Stockfish tool-calling systems was sparse and noisy. No mature canonical repo emerged for “chess engine LLM tool calling.”
- Useful chess implementation hits existed, but mostly as isolated chess-analysis tools, not robust benchmark/training pipelines.
- Stronger evidence came from general tool-calling systems and benchmarks:
  - `sierra-research/tau-bench`
  - `sierra-research/tau2-bench`
  - `OpenBMB/ToolBench`
  - `THUNLP-MT/StableToolBench`
  - `ShishirPatil/gorilla`
  - `IBM/vakra`
  - `SeraphimSerapis/tool-eval-bench`
  - `MikeVeerman/tool-calling-benchmark`

### Library / official docs

- `python-chess` supports legal move generation, SAN/UCI parsing, push/pop state transitions, and UCI engine communication.
- vLLM supports structured outputs via JSON schema and OpenAI-compatible clients.
- llama.cpp supports GBNF grammar-constrained generation and JSON-schema-to-grammar workflows.
- OpenAI docs support strict structured outputs / function calling with `strict: true`, required fields, and `additionalProperties: false`.

## Audit of prior pipeline

### 1. “Contracts first” was necessary but incomplete

Prior claim: freeze tool contracts early.

Audit: correct, but insufficient. Tool contracts alone do not define correctness. τ-bench-style systems define a full domain environment: policy, tools, tasks, user simulator, and expected state/result behavior.

Correction: freeze **domain contract bundle**:

1. tool schemas
2. result schemas
3. error schemas
4. board/session state model
5. policy text
6. task/scenario spec
7. evaluator rules
8. trace envelope
9. version IDs for all above

Gate: no dataset generation until contract bundle passes schema validation and replay smoke tests.

### 2. “Engine oracle” must become executable environment, not backend only

Prior claim: backend owns chess truth.

Audit: correct but underspecified. Evidence from VAKRA emphasizes executable, verifiable evaluation where trajectories replay against live local tools. StableToolBench exists because real/external APIs create instability; chess can avoid that by making Stockfish/python-chess local and deterministic.

Correction: implement **local chess environment** with:

- board database/session state
- legal move generator
- engine adapter
- deterministic analysis settings
- scenario initializers
- task expected outcomes
- trace replayer
- evaluator

Gate: every scenario can run without LLM and produce expected oracle traces.

### 3. “Dataset factory after eval” order needs sharper split

Prior pipeline placed eval before dataset factory in final summary but also described dataset phase before eval. That inconsistency is real.

Correct order:

1. environment contract
2. oracle implementation
3. evaluator implementation
4. seed scenario pack
5. baseline prompt/model eval
6. trace generation
7. dataset filtering/replay
8. training
9. held-out eval

Dataset is downstream of evaluator. SFT examples are accepted only if they replay and grade cleanly.

### 4. `ask_chessbot` should be removed from core tool set

Prior claim: restrict or remove `ask_chessbot`.

Audit: strengthen to remove from MVP. A fallback meta-tool undermines oracle authority and contaminates eval: model can route hard chess truth to a non-oracle explanatory channel.

Correction:

- Remove `ask_chessbot` from state/engine MVP.
- Use `explain_concept` only for non-position-specific chess education, if needed.
- Enforce policy: any claim about current board, legal moves, eval, tactic, best move, mate, material, or threat requires oracle-backed tool result.

Gate: evaluator fails any position-specific chess claim not grounded in tool output.

### 5. Mode 1 / Mode 2 remains useful, but should be replaced by channel isolation

Prior claim: router/narrator split physically prevents Mode 2 tool calls.

Audit: correct. Make this mandatory, not optional. Tool-eval-bench and BFCL-style failures show format sensitivity and multi-turn orchestration matter. Prompt-only mode discipline is not enough.

Correction:

- Router has tool-call channel.
- Narrator has no tool-call channel.
- Runtime never executes tool-like text from narrator.
- Narrator input must include only validated tool result, not raw backend logs.

Gate: executable tool calls are impossible in narration phase by API/channel design.

### 6. Evaluation gates were too arbitrary

Prior numbers like 95% route accuracy and 90% trajectory success were placeholders, not evidence-backed thresholds.

Correction: use staged gates:

- **Blocker gates:** binary, must be 100%.
  - schema-valid executable calls
  - no unknown tool names
  - no invalid state mutation
  - no invented tool result in final answer
  - no Mode 2 executable call
  - all state transitions replay
- **Quality gates:** improve by version and task class.
  - route accuracy
  - trajectory success
  - final answer groundedness
  - helpfulness
  - latency/cost

Set first release thresholds after baseline distribution, not before. Initial target can be “zero criticals + statistically better than prompt-only baseline.”

### 7. ToolBench is useful but warns against external API instability

ToolBench uses many APIs and evaluator-based pass/win rates. StableToolBench exists because unstable APIs and query solvability introduce randomness.

Correction for chess:

- Do not use external web/API dependencies in core eval.
- Use local deterministic tools.
- Prefer programmatic checks over LLM judges.
- Use LLM judge only for coaching style/helpfulness after correctness passes.

Gate: correctness score independent of LLM judge.

### 8. Need policy layer, not only tools

τ-bench domains include policy, tools, and tasks. Chess assistant needs explicit policy:

- what assistant may answer without tools
- what requires legal-move tool
- what requires engine tool
- how to handle ambiguous SAN
- how to handle casual chess education
- how to handle user requests to guess
- how to handle engine timeout
- how to present uncertainty

Gate: evaluator checks policy adherence separately from tool correctness.

### 9. Need user simulator / scenario generator, but bounded

τ-bench uses user simulators. For chess, free LLM user simulation can generate invalid/noisy tasks.

Correction:

- Use templated scenario generator first.
- Use LLM user simulator only to paraphrase or adversarially perturb validated scenarios.
- Every generated scenario must compile against board state and expected tool route.

Gate: scenario enters eval set only after oracle compile + human/audit approval for expected route.

### 10. Need contamination and split discipline

Gorilla/BFCL roadmap calls out contamination metrics. Chess tasks are easy to leak because positions repeat.

Correction:

Split by:

- game source
- position family
- tactical motif
- opening family
- scenario template
- paraphrase cluster

Never random-split individual examples only.

Gate: train/dev/test have no shared exact FEN, no shared source game, and no shared near-duplicate template unless explicitly measuring paraphrase robustness.

## Revised pipeline

### Phase 0 — Search-first prior art ledger

Artifact: `literature/NOTES.md` + source table.

Must include:

- official docs for chess backend and inference backend
- tool-calling benchmark repos
- function-calling eval methods
- known benchmark instability lessons
- chess-specific implementation search results, including negative result if no canonical project found

Gate: every pipeline design claim links to source or is labeled assumption.

### Phase 1 — Domain contract bundle

Artifact: versioned domain spec.

Contains:

- tool schemas
- result schemas
- error schemas
- state model
- trace schema
- policy
- scenario schema
- evaluator rubric

Gate: JSON schemas validate; unknown fields rejected; every state-mutating tool has idempotency key.

### Phase 2 — Local executable chess environment

Artifact: environment runner.

Components:

- python-chess board/session core
- Stockfish/UCI adapter
- deterministic engine settings
- task initializer
- trace recorder
- replay runner

Gate: no LLM needed to run scenario oracle traces.

### Phase 3 — Policy-first router/narrator runtime

Artifact: baseline runtime harness.

Design:

- router emits strict structured tool call or direct answer class
- backend validates and executes
- narrator receives validated tool result and cannot call tools

Gate: narration channel has no executable tools.

### Phase 4 — Evaluator before dataset

Artifact: evaluator CLI/report.

Checks:

- schema validity
- route correctness
- policy adherence
- state transition correctness
- exact/normalized tool result match
- final answer groundedness
- trajectory success
- safety/adversarial behavior

Gate: evaluator can grade handcrafted golden traces and injected bad traces.

### Phase 5 — Scenario packs

Artifact: versioned eval packs.

Packs:

- legal move
- illegal move
- ambiguous move
- best move
- eval
- review move
- threats
- undo/state
- timeout/error
- adversarial/prompt injection
- educational no-tool
- long trajectory

Gate: every scenario has expected route class and oracle-checkable outcome.

### Phase 6 — Baseline model evaluation

Artifact: baseline report.

Evaluate:

- prompt-only
- grammar/schema constrained
- router/narrator split
- maybe several small local models

Gate: choose training target from measured failure buckets.

### Phase 7 — Trace/data generation

Artifact: replayable SFT candidate pool.

Generate only from validated scenario packs and oracle traces. Include hard negatives and no-tool cases.

Gate: every example replays; every assistant claim grounded; no train/eval leakage.

### Phase 8 — Training ladder

Artifact: checkpoints + eval reports.

Order:

1. router SFT for route/tool args
2. narrator SFT for grounded explanation
3. hard-negative SFT
4. preference/rejection sampling only after correctness gates
5. RL-style only after deterministic simulator matures

Gate: checkpoint accepted only if blocker gates stay perfect and quality gates improve by bucket.

### Phase 9 — Release harness

Artifact: release report.

Release requires:

- zero blocker failures
- trajectory replay pass
- source/version manifest
- regression comparison vs previous model/runtime
- rollback plan

Gate: no demo-based release.

## Corrected core principle

Old principle:

> Model chooses and explains. Backend knows and mutates. Runtime constrains and audits. Eval decides release.

Revised principle:

> Environment defines truth. Runtime enforces channels. Evaluator grades trajectories. Dataset is only replayable evidence. Model learns routing and grounded narration, never chess authority.

## Remaining assumptions requiring more research

1. Best deterministic Stockfish settings for stable evaluation across machines.
2. How to normalize engine centipawn/mate/PV outputs without overfitting to exact depth noise.
3. Whether local small models route better with native tool-call fine-tuning or pure JSON grammar plus router prompt.
4. Best way to evaluate coaching helpfulness after correctness gates pass.
5. Whether FEN-blind product constraint should apply to internal traces or only user-visible/model-visible context.

## Sources

- https://github.com/sierra-research/tau-bench
- https://github.com/sierra-research/tau2-bench
- https://github.com/OpenBMB/ToolBench
- https://github.com/THUNLP-MT/StableToolBench
- https://github.com/ShishirPatil/gorilla
- https://github.com/IBM/vakra
- https://github.com/SeraphimSerapis/tool-eval-bench
- https://github.com/MikeVeerman/tool-calling-benchmark
- https://github.com/niklasf/python-chess
- https://docs.vllm.ai/en/latest/features/structured_outputs
- https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md
- https://developers.openai.com/api/docs/assistants/tools/function-calling
