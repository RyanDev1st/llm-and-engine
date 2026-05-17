## Why

LLM phase done; next bottleneck chess engine behavior quality and reliability for tool-call execution path. Need deterministic, testable engine contract now so routed tool calls return stable, high-signal results for narration phase.

## What Changes

- Add chess engine capability spec covering board state lifecycle, move legality, evaluation, best-move, undo, and error classes.
- Define deterministic request/response schemas for all engine tools consumed by LLM runtime.
- Define runtime invariants for state consistency across multi-turn sessions.
- Add acceptance scenarios for engine_unavailable, timeout, invalid position, and invalid move handling.
- Define replay-grade tolerances for non-deterministic eval families.

## Capabilities

### New Capabilities
- `chess-engine`: Deterministic chess engine interface and behavior contract for tool-calling runtime.

### Modified Capabilities
- `LLM`: Clarify dependency contract between router/narrator flow and engine error/result classes.

## Impact

- Affects tool backend integration layer in `src/`.
- Affects dataset replay validation families and red-team probes.
- Affects end-to-end tool-call execution reliability and user-facing narration grounding.
