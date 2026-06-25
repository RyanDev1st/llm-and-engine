# docs/ — project documentation index

Only **current, in-use** docs live here. Organized by **lifecycle**, not topic. Root plan files (`implementation.md`, `handoff.md`) live at the repo root, not here. **Lessons/experience → memory** (one-fact files), never a docs folder.

| Bucket | Holds | Lifecycle |
| --- | --- | --- |
| [`reference/`](reference/) | How the system works **now** — architecture, contracts, designs | **Mutable in place**; update when the system changes |
| [`findings/`](findings/) | Dated investigations — audits, triage, inspections, eval reports | **Immutable**; supersede with a newer-dated file, archive the old to `legacy/` |
| [`adr/`](adr/) | Architecture Decision Records — the **why** behind a non-obvious choice | **Immutable**; a reversal is a new ADR that supersedes |
| [`legacy/`](legacy/) | Superseded/retired docs | Archive — never edited, never deleted |

A tool that writes a report MUST write a fresh `docs/findings/YYYY-MM-DD-<topic>.md` — never overwrite an archived or reference doc.

## reference/ — how it works now

| Doc | What it is |
| --- | --- |
| [reference/harness-system-overview.md](reference/harness-system-overview.md) | **Peer-review packet** — single self-contained overview of the whole system (product thesis, model, contract, loop, corpus, robustness, latest eval evidence) + an explicit known-gaps/open-questions list for an external reviewer. Start here for a top-down read |
| [reference/sft-corpus-generation.md](reference/sft-corpus-generation.md) | Plain-English account of how the v1.2 training data is built — factory model, the 20 cards + 25 slices, combinatorial assembly, grounding, generalization, quality gate, split. Source doc for the report/talk |
| [reference/glossary.md](reference/glossary.md) | Canonical terminology — skill, tool, plugin, skill index, tool manifest, hook gate, etc. |
| [reference/harness-architecture.md](reference/harness-architecture.md) | The agent harness: deterministic routing layer, tools, skills, plugins, end-to-end turn trace |
| [reference/harness-strengthening.md](reference/harness-strengthening.md) | The 3-pillar strengthening (memory / routing / thinking): memory tiers + write gate, prefix KV reuse, live-think streaming — what changed per pillar + how to test |
| [reference/2026-05-23-chess-coach-sft-design.md](reference/2026-05-23-chess-coach-sft-design.md) | Foundational SFT/agent design (the contract) |
| [reference/2026-06-12-coverage-reliability-design.md](reference/2026-06-12-coverage-reliability-design.md) | Coverage + grounding reliability layer (shipped) |
| [reference/2026-06-09-context-window-system.md](reference/2026-06-09-context-window-system.md) | Session-memory window: token budget + eviction |
| [reference/2026-06-11-stockfish-uci-contract.md](reference/2026-06-11-stockfish-uci-contract.md) | Stockfish UCI interface contract |

## findings/ — dated investigations (immutable)

| Doc | What it is |
| --- | --- |
| [findings/2026-06-25-answer-quality-serve-fixes-and-corpus-finding.md](findings/2026-06-25-answer-quality-serve-fixes-and-corpus-finding.md) | The first full report run's metrics looked strong (OOD completion 91.7%) but the FINAL ANSWERS were often unhelpful. Classifies the 14 live showcase turns into SERVE bugs (fixed, CPU-verified: `move e4` arg coercion, the `Coach: <` markup-fragment whiff, raw-result-echo strip) vs a DATA problem. Corpus-finals analysis (72,329 rows): closer monoculture (70% end with a question; 100% in chess slices + the 18k V1_O) + **V1_N process-narration finals 71.5%** ("I ran the helper, then loaded chess-coach" instead of answering) = the narrate-and-deflect failure, in the data. Scopes a TARGETED v5 (rewrite V1_N answer-first + break the closer monoculture); NOT the harness |
| [findings/2026-06-25-serve-latency-investigation.md](findings/2026-06-25-serve-latency-investigation.md) | Serve latency root-cause: it's DECODE-bound (tokens × steps × T4 tok/s), not prefill-bound. KV reuse only saves prefill (~tenths of a sec) and isn't even on the live streaming path; Stockfish depth isn't the lever (all 1353 corpus calls pass explicit depth; decode dominates). Real levers: fewer decode steps (thin mode + the verify-probe gating shipped here + the PR#20 bug fixes that cut error-recovery loops) and a faster-decoding model (GGUF Q4_0 ~2.4×). Don't cut the `<think>` trace. Measure via the [gen] trace |
| [findings/2026-06-24-harness-live-vs-benchmark-gap.md](findings/2026-06-24-harness-live-vs-benchmark-gap.md) | Root-cause of "live ≠ the 96% benchmark": three different prompts in play (train / benchmark / live). The benchmark scores a prompt the server never sends (bare `build_system`, single-turn, 48-tok, first-action). Confirmed off-distribution divergence: the served `LIVE BOARD` system-prompt injection (0/2731 training rows; contradicts the trained "board is hidden" rule). Both user hypotheses true — "model struggles in the harness" primary (off-distribution prompt), "stops too soon" secondary (rescue-layer hidden gens). Ships `CHESS_BOARD_HOOK` + `CHESS_THIN_HARNESS` flags + a 2×2 GPU A/B plan via `eval_completion` |
| [findings/2026-06-24-harness-vs-claude-code-codex.md](findings/2026-06-24-harness-vs-claude-code-codex.md) | Feature-by-feature side-by-side of our serve harness vs the thin top-harness principle (Claude Code/Codex), under the constraint "keep the trained contract exactly". Verdict: contract + plumbing = KEEP (already match); the deterministic rescue layer (coverage force-routing, per-turn self-verify probe, output rewriting, deflection detectors) = over-engineered for the weak E2B, off-distribution, the E4B's bare contract routes at 96% — thin it, staged + benchmark-gated. One real GAP: grammar-constrained decoding to replace regex call-repair |
| [findings/2026-06-22-benchmark-rerun-and-harness-refinement.md](findings/2026-06-22-benchmark-rerun-and-harness-refinement.md) | Kaggle rerun read (version trend, native-mode fair test adapter 80.2% vs base 41.9% on hard slices, degenerate e2b condition) → CORRECTS the slice-G "fast-mode artifact" claim (it's a real tool-as-skill verb confusion) → ships verb-coercion in extract_call so the loop is robust without depending on frozen-model recovery from an off-distribution corrector |
| [findings/2026-06-22-peer-review-gap-verification.md](findings/2026-06-22-peer-review-gap-verification.md) | Verifies an external peer review's 6 harness gaps (all TRUE, file:line evidence) — chess routing force-fires on OOD prompts, first-action-only eval, n=20 stress, GGUF decode drift, frontend reconcile, engine-error mislabel — + a prioritized remediation plan (P0 correctness → P1 proof/fidelity → P2 UI) |
| [findings/2026-06-21-routing-benchmark-interpretation.md](findings/2026-06-21-routing-benchmark-interpretation.md) | 3-condition routing ablation read (val 96.4% verb / 78.3% macro-prec; harness vs SFT-weights deltas); slice-G/H = over-specialization not label bug; drove the symmetric tool-as-skill corrective-error fix |
| [findings/2026-06-20-harness-serve-gap-audit.md](findings/2026-06-20-harness-serve-gap-audit.md) | Serve-loop gap audit grounded vs Anthropic/OpenAI loops; 2 PROVEN gaps (`<think>`/`<goal>` reply leak; direct-answer-as-plan-panel) + G3 mode-wiring; report-only, fixes pending |
| [findings/2026-06-14-v1.2-corpus-audit.md](findings/2026-06-14-v1.2-corpus-audit.md) | Training-readiness audit (GATE PASS; loss mask, fast/think/auto, V1_R Stage-0); pre-Kaggle GO |
| [findings/2026-06-13-v1.2-random-sample-inspection.md](findings/2026-06-13-v1.2-random-sample-inspection.md) | Truly-random v1.2 train sample (10 rows/slice) for human inspection; regen with `scripts/make_sample_doc.py` |
| [findings/2026-06-12-model-audit.md](findings/2026-06-12-model-audit.md) | Current model behavior audit |
| [findings/2026-06-12-serve-audit-readiness.md](findings/2026-06-12-serve-audit-readiness.md) | Serve-side readiness audit |
| [findings/2026-06-12-gguf-vs-hf-triage.md](findings/2026-06-12-gguf-vs-hf-triage.md) | GGUF vs HF serving triage |
| [findings/2026-06-06-v1.2-dataset-alignment-audit.md](findings/2026-06-06-v1.2-dataset-alignment-audit.md) | Decision basis for Option B (cited by `implementation.md`) |

## adr/ — decisions (the why)

| ADR | Decision |
| --- | --- |
| [adr/0001-episodic-tool-memory.md](adr/0001-episodic-tool-memory.md) | Add a 4th memory tier — episodic, global, file-backed, flag-gated — that learns a tool's correct usage from a turn's error→fix and recalls it for similar later requests. The system self-learns with the model frozen; persists across runs/machines via a synced dir. Rejects retraining + global-regex arg-fillers |

Add `adr/NNNN-short-title.md` when a non-obvious choice is made (sequential number). See [adr/README.md](adr/README.md).

---
`screenshots/` holds UI captures referenced by docs.
