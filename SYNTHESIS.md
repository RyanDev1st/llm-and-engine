# Synthesis — LLM tool calling for AI engines

**Status:** Draft — merged search-first and adversarial pipeline audits.  
**Owner:** chatgpt-main  

## Executive summary

The strongest pattern across tool-calling benchmarks is environment-first design: policy, tools, state, tasks, replay, and evaluator define correctness before training data exists. For the chess assistant, this means the model should learn routing and grounded narration while python-chess/Stockfish and a deterministic local environment own chess truth. The prior v3 spec has useful foundations, but its dataset-first order, unified prompt, and broad `ask_chessbot` fallback make correctness evaluation too weak. Adversarial review tightened the corrected design further: “deterministic engine,” “grounded answer,” “ambiguous move,” and “evaluator pass” must each have explicit counterexample tests, or the pipeline can still certify a brittle model. Ralph loop 2 converted the remaining abstract gates into minimum schema skeletons, an evaluator mutant table, a seed scenario coverage matrix, a baseline model/runtime matrix, and explicit release pass/fail semantics. The final direction is executable environment first, concrete contract bundle second, evaluator mutation suite third, baseline eval fourth, and replayable data only afterward.

## Consensus (≥2 independent sources or strong industry convergence)

1. **Tool contracts alone are not enough.** Benchmarks such as tau-bench/tau2, VAKRA, and local deterministic tool-calling benchmarks frame correctness around an executable environment: policy, tools, tasks, state, and expected outcomes.
2. **Local deterministic tools are preferable for correctness eval.** StableToolBench exists because real/external APIs make ToolBench-style evaluation unstable. Chess can avoid that problem by using python-chess and local Stockfish.
3. **Replayable trajectories matter more than isolated calls.** Multi-turn tool use fails through state drift, wrong recovery, invalid tool sequencing, and ungrounded final answers; evaluator must grade trajectories, not only single function calls.
4. **Constrained decoding improves format adherence but not semantic correctness.** vLLM structured outputs, llama.cpp grammars, and strict schemas can prevent malformed calls, but they do not prove the model chose the right tool or stayed policy-grounded.
5. **Critical gates should be binary.** Unknown tools, schema-invalid calls, illegal state mutation, invented tool results, and executable post-tool narration should be blocker failures with a 100% pass requirement.
6. **Evaluator gates need mutation tests.** A gate is not trustworthy until one-bug-per-trace mutants prove it catches the exact failure it claims to block.
7. **Engine reproducibility must be contractual.** Stockfish version, NNUE/EvalFile, Threads, Hash, MultiPV, Clear Hash policy, and search limit type must be recorded for release-critical packs.
8. **Abstract gates must be made executable before implementation.** Contract bundles need minimum schema skeletons, scenario packs need coverage matrices, baselines need a model/runtime matrix, and releases need field-level pass/fail semantics.

## Contradictions or unsettled debates

| Topic | Position A | Position B | Notes |
|-------|------------|------------|-------|
| Unified prompt vs router/narrator | Unified prompt keeps data simple and trains one record shape. | Router/narrator split gives runtime channel isolation and prevents Mode 2 executable calls. | Correctness-first design should make split runtime mandatory, while keeping unified prompt only as a baseline. |
| Exact move emission vs move-resolution step | Router emits explicit SAN/UCI directly from user text. | Ambiguous natural-language move requests need resolver/clarification before `move`. | Direct SAN emission is brittle for phrases like “take with the knight” or “castle.” |
| Tool-call fine-tuning vs JSON grammar router | Native tool-call fine-tuning may improve learned routing behavior. | JSON grammar/schema router may be easier for small local models and runtimes. | Needs measured baseline by target model size. |
| Exact engine-output replay vs normalized grading | Exact replay gives crisp deterministic checks. | Engine depth, hardware, and version noise can make exact score/PV matching brittle. | Use fixed settings plus normalized bins for evaluator comparisons; preserve raw outputs in traces. |
| FEN-blind traces | FEN should never appear because product is FEN-blind. | Internal traces need FEN/hash for replay and leakage detection. | Keep user/model-visible context FEN-blind; allow internal evaluator metadata if explicitly isolated. |
| Coaching helpfulness eval | LLM judge can assess style and usefulness. | LLM judges should not determine chess correctness. | Use programmatic correctness first; judge style only after blocker gates pass. |

## Gaps in literature or practice

- No mature canonical chess-specific LLM+Stockfish tool-calling benchmark/training pipeline emerged from search.
- Deterministic Stockfish settings for cross-machine reproducibility need deeper validation.
- Centipawn, mate, and PV normalization policy must be defined before final evaluator thresholds.
- Small-model routing behavior for native tool calls versus grammar-constrained JSON remains empirical.
- The boundary between internal replay metadata and model-visible FEN-blind context needs explicit product policy.
- Groundedness detection needs a claim taxonomy, or implicit board claims can slip through evaluator checks.
- Long-horizon state drift across castling rights, en passant, promotion, and undo remains easy to under-test.
- Concrete JSON Schema files, seed FEN/source-game fixtures, exact local model names, hardware budget, and numeric quality floors remain implementation-phase decisions.

## Recommendations (research / design only — no code commitment)

1. Treat the first artifact as a versioned executable chess environment, not an SFT dataset.
2. Freeze a domain contract bundle: tool schemas, result schemas, error schemas, state model, policy, scenario schema, evaluator rules, trace envelope, and version manifest.
3. Remove `ask_chessbot` from the correctness MVP; use an isolated no-position education path later if needed.
4. Use API-level router/narrator channel isolation; never depend on prompt-only Mode 2 discipline for production.
5. Build evaluator before data generation; require golden traces and injected bad traces to grade correctly.
6. Generate SFT only from validated scenario packs and accepted oracle traces.
7. Set quality thresholds after baseline distributions, but keep blocker gates at 100%.
8. Split train/dev/test by source game, exact FEN, position family, motif, opening family, scenario template, and paraphrase cluster.
9. Add engine reproducibility profile before using Stockfish outputs as release gates.
10. Add move-resolution policy for underspecified natural language before `move` execution.
11. Add final-answer claim taxonomy and map each claim type to required evidence.
12. Add long-game state drift, special-move, and prompt-injection scenario packs.
13. Require minimum schema skeletons before contract bundle v0 is accepted.
14. Require evaluator mutant table and seed scenario coverage matrix before baseline measurement.
15. Require baseline model/runtime matrix and measurement protocol before training target selection.
16. Require release manifest pass/fail semantics before any release pack can be called shippable.

## Mapping to chess assistant v3 spec

- **Tool surface / routing:** Keep core position tools (`move`, `eval`, `best_move`, `review_move`, `threats`, `legal_moves`, `undo`, `list_pieces`) but remove `ask_chessbot` from correctness MVP because it weakens route accountability.
- **Validation / replay:** Replace dataset-validation-first mindset with environment/evaluator-first validation. Every scenario should compile and replay before becoming training data.
- **Mode discipline (post-tool narration):** Replace unified-prompt discipline with runtime channel isolation. Router can call tools; narrator cannot.
- **Dataset order:** Move SFT generation downstream of evaluator, seed scenario packs, and baseline model evaluation.
- **Release criterion:** Ship only versioned release packs with zero blocker failures and replayed held-out trajectories.

## Appendix — index of agent reports

| File | Agent | One-line takeaway |
|------|-------|-------------------|
| `findings/ralph_loop_2_results.md` | chatgpt-main | Second Ralph loop made abstract contract, scenario, baseline, and release gates concrete enough to audit. |
| `findings/ralph_loop_1_results.md` | chatgpt-main | First Ralph loop tightened reproducibility, evaluator mutation, move resolution, and state-drift requirements. |
| `findings/adversarial_pipeline_audit.md` | chatgpt-main | Corrected spec still needed engine reproducibility, move-resolution, claim-taxonomy, evaluator mutation, and state-drift gates. |
| `findings/search_first_pipeline_audit.md` | chatgpt-main | Correct pipeline starts with executable environment and evaluator, not dataset generation. |
