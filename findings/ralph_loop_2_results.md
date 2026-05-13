# Ralph loop 2: concrete contract and release-gate tightening

**Date:** 2026-05-13  
**Mode:** attack remaining abstraction until spec becomes executable  
**Scope:** research/spec only  

## What broke under attack

Ralph loop 1 fixed the major architecture direction, but four parts could still hide implementation bugs behind good-sounding language:

1. “Contract bundle” did not force concrete schema fields.
2. “Scenario packs” did not name seed coverage rows.
3. “Baseline report” did not define target model/runtime matrix or measurement protocol.
4. “Release manifest” did not say which missing fields or failed checks block release.

## Corrections made

1. Added minimum schema skeleton requirements to the design spec:
   - tool call envelope
   - tool result envelope
   - error envelope
   - session state snapshot
   - scenario schema
   - trace envelope
   - version manifest

2. Added explicit evaluator mutant table:
   - unknown tool
   - missing required arg
   - wrong arg value
   - illegal state mutation
   - read-only mutation
   - invented tool result
   - ungrounded final answer
   - FEN leakage
   - prompt-injection compliance
   - split contamination

3. Added seed scenario coverage matrix:
   - legal/illegal moves
   - ambiguous and underspecified moves
   - eval, best move, review, threats, undo, list pieces
   - castling, en passant, promotion, mate/stalemate/draw
   - prompt injection, timeout/recovery, long-game state drift
   - educational no-position path

4. Added baseline model/runtime matrix and measurement protocol:
   - prompt-only unified baseline
   - grammar/schema-constrained router
   - router/narrator split
   - local candidate small models
   - per-bucket metrics, identical scenario packs, fixed engine profile, bootstrap confidence intervals, and qualitative review for failure clusters

5. Added release manifest pass/fail semantics:
   - missing required manifest field blocks release
   - any blocker failure blocks release
   - evaluator mutant-suite failure blocks release
   - replay failure blocks release
   - contamination blocks release
   - regression above threshold blocks release
   - quality buckets below floor require explicit non-release or documented limitation

## New score

Checklist score moved from **6/10** to target **10/10** for the current Ralph loop gap list.

## Remaining attack surface for next loop

1. Schema skeletons still need actual JSON Schema files before implementation.
2. Scenario seeds need concrete FEN/source-game examples once product policy permits internal replay metadata.
3. Baseline matrix still needs exact model names and hardware budget after target deployment constraints are known.
4. Release quality floors need numeric values from baseline distributions.
5. Education path remains a product choice: direct answer from model weights, KB-backed `explain_concept`, or excluded from MVP.
