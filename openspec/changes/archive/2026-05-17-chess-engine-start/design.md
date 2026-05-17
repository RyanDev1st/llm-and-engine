## Context

LLM capability now defined and archived. Next phase requires deterministic chess engine behavior so tool calls return reliable outputs for narration grounding. Current project has replay gates and red-team loops; engine contract must align with those gates and expose stable error classes.

## Goals / Non-Goals

**Goals:**
- Define deterministic engine interface for tool-calling runtime.
- Define state lifecycle invariants for multi-turn sessions.
- Define error taxonomy and timeout behavior usable by narrator.
- Define tolerance policy for non-deterministic eval families.

**Non-Goals:**
- Model training changes.
- UI concerns.
- Opening-book strength optimization beyond contract minima.

## Decisions

1. Single engine-session state object keyed per conversation.
   - Rationale: preserves board continuity across turns and supports undo deterministically.
   - Alternative rejected: stateless FEN-only calls on each turn (too error-prone for long sessions).

2. Tool contracts use strict JSON schemas with explicit error envelope.
   - Rationale: replay validator and narrator need machine-checkable outcomes.
   - Alternative rejected: freeform text tool output (breaks deterministic validation).

3. Eval family split into exact and tolerance checks.
   - Rationale: move legality/state ops must be exact; engine eval varies by depth/time.
   - Alternative rejected: all-exact policy (false failures under acceptable variance).

4. Timeout and engine_unavailable normalized to canonical error codes.
   - Rationale: narrator behavior depends on stable error class mapping.
   - Alternative rejected: backend-specific errors passed through directly.

## Risks / Trade-offs

- Engine determinism vs strength tuning conflict → lock test profile for validation runs.
- Session state corruption across concurrent calls → enforce single-writer per session.
- Timeout thresholds too aggressive → tune with replay histogram and widen only where needed.
- Error over-normalization hides debugging detail → return normalized code + backend detail field.

## Migration Plan

1. Add chess engine spec and LLM delta spec.
2. Implement engine tool schemas and adapter in `src/`.
3. Add validator updates for new error/tolerance policies.
4. Run replay + red-team + audit gates.
5. If gates fail, patch via targeted regeneration and rerun.

Rollback: disable engine-backed routing path and fall back to current stable dataset evaluation path.

## Open Questions

- Final timeout budget per tool family (`best_move`, `eval`) for edge hardware profile.
- Whether `review_move` should be tolerance family or mixed exact+tolerance on subfields.
