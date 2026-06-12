# docs/ — active project documentation

Only **current, in-use** docs live here. Superseded ones move to [`legacy/`](legacy/).
Root plan files (`implementation.md`, `handoff.md`) live at the repo root, not here.

## Durable references (no date — update in place)

| Doc | What it is |
| --- | --- |
| [harness-architecture.md](harness-architecture.md) | The agent harness: deterministic routing layer, tools, skills, plugins, end-to-end turn trace |
| [superpowers/specs/2026-05-23-chess-coach-sft-design.md](superpowers/specs/2026-05-23-chess-coach-sft-design.md) | Foundational SFT/agent design (the contract) |
| [superpowers/specs/2026-06-12-coverage-reliability-design.md](superpowers/specs/2026-06-12-coverage-reliability-design.md) | Coverage + grounding reliability layer (shipped) |

## System / contract docs

| Doc | What it is |
| --- | --- |
| [2026-06-09-context-window-system.md](2026-06-09-context-window-system.md) | Session-memory window: token budget + eviction |
| [2026-06-11-stockfish-uci-contract.md](2026-06-11-stockfish-uci-contract.md) | Stockfish UCI interface contract |
| [2026-06-06-v1.2-dataset-alignment-audit.md](2026-06-06-v1.2-dataset-alignment-audit.md) | Decision basis for Option B (cited by `implementation.md`) |

## Latest audits / triage (dated; supersede with a newer date, archive the old one)

| Doc | What it is |
| --- | --- |
| [2026-06-12-model-audit.md](2026-06-12-model-audit.md) | Current model behavior audit |
| [2026-06-12-serve-audit-readiness.md](2026-06-12-serve-audit-readiness.md) | Serve-side readiness audit |
| [2026-06-12-gguf-vs-hf-triage.md](2026-06-12-gguf-vs-hf-triage.md) | GGUF vs HF serving triage |

`screenshots/` holds UI captures referenced by docs.
