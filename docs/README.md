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
| [findings/2026-06-20-harness-serve-gap-audit.md](findings/2026-06-20-harness-serve-gap-audit.md) | Serve-loop gap audit grounded vs Anthropic/OpenAI loops; 2 PROVEN gaps (`<think>`/`<goal>` reply leak; direct-answer-as-plan-panel) + G3 mode-wiring; report-only, fixes pending |
| [findings/2026-06-14-v1.2-corpus-audit.md](findings/2026-06-14-v1.2-corpus-audit.md) | Training-readiness audit (GATE PASS; loss mask, fast/think/auto, V1_R Stage-0); pre-Kaggle GO |
| [findings/2026-06-13-v1.2-random-sample-inspection.md](findings/2026-06-13-v1.2-random-sample-inspection.md) | Truly-random v1.2 train sample (10 rows/slice) for human inspection; regen with `scripts/make_sample_doc.py` |
| [findings/2026-06-12-model-audit.md](findings/2026-06-12-model-audit.md) | Current model behavior audit |
| [findings/2026-06-12-serve-audit-readiness.md](findings/2026-06-12-serve-audit-readiness.md) | Serve-side readiness audit |
| [findings/2026-06-12-gguf-vs-hf-triage.md](findings/2026-06-12-gguf-vs-hf-triage.md) | GGUF vs HF serving triage |
| [findings/2026-06-06-v1.2-dataset-alignment-audit.md](findings/2026-06-06-v1.2-dataset-alignment-audit.md) | Decision basis for Option B (cited by `implementation.md`) |

## adr/ — decisions (the why)

_None yet._ Add `adr/NNNN-short-title.md` when a non-obvious choice is made (sequential number). See [adr/README.md](adr/README.md).

---
`screenshots/` holds UI captures referenced by docs.
