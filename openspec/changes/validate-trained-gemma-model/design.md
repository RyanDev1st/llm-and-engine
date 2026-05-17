## Context

Gemma training is intended to produce a chess assistant that can choose tools, emit valid JSON arguments, and narrate only after tool results. Prior local attempts show trained artifacts can exist while full Gemma4 CUDA training may fail on 8GB VRAM, so validation must separate artifact readiness from hardware-limited training.

## Goals / Non-Goals

**Goals:**
- Validate any named trained Gemma artifact with deterministic chess assistant scenarios.
- Measure tool-call correctness, JSON-schema validity, illegal narration before tool execution, and post-tool answer quality.
- Produce a findings report that records commands, artifacts, metrics, and blocking failures.
- Keep tests runnable without committing model weights.

**Non-Goals:**
- Guarantee Gemma4 26B training fits on 8GB VRAM.
- Replace chess engine evaluation with mocked production behavior.
- Commit local runtime caches, downloaded models, or generated weights.

## Decisions

- Use an artifact-driven validation entry point. Rationale: training and validation fail for different reasons; a saved artifact can be validated even when fresh training is blocked by VRAM. Alternative considered: always train before validation; rejected because hardware OOM would hide validation regressions.
- Validate tool calls with schema checks before semantic scoring. Rationale: malformed calls cannot be safely executed. Alternative considered: score generated text only; rejected because tool-calling product quality depends on machine-readable calls.
- Use deterministic fixture prompts plus optional real-engine checks. Rationale: fixed prompts make regression tests stable, while engine-backed checks prove integration behavior. Alternative considered: only live engine evaluation; rejected because failures become harder to isolate.
- Write evidence to `legacy/findings/`. Rationale: project rules require durable reports for findings and milestone status.

## Risks / Trade-offs

- Hardware-limited Gemma runs may remain blocked -> validation must report OOM separately from model-quality failure.
- Small deterministic set may miss broad behavior issues -> include extensible scenario files and aggregate metrics.
- Real engine dependency may be unavailable -> mark integration checks blocked, not passed.

## Migration Plan

1. Add validation harness and tests behind explicit artifact path arguments.
2. Run deterministic validation against available trained artifacts.
3. Publish findings report with pass/fail metrics and blockers.
4. Promote only artifacts that meet thresholds.

## Open Questions

- Which trained Gemma artifact path should be first validation target if multiple exist?
- What minimum pass thresholds should gate promotion after first baseline run?
