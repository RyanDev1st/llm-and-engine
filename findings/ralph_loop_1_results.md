# Ralph loop 1: adversarial tightening results

**Date:** 2026-05-13  
**Mode:** attack own points until stronger  
**Scope:** research/spec only  

## What broke under attack

The corrected search-first pipeline still had weak language around “deterministic,” “grounded,” “ambiguous,” and “evaluator pass.” Each sounded right but could hide bugs.

## Corrections made

1. Added engine reproducibility profile to design spec:
   - Stockfish binary/version hash
   - NNUE/EvalFile hash
   - `Threads=1`
   - fixed `Hash`
   - `Clear Hash` policy
   - fixed `MultiPV`
   - fixed `depth`/`nodes` for release-critical packs
   - full engine trace metadata

2. Tightened tool behavior:
   - `review_move` must use copied positions, not live pop/re-push mutation.
   - `move` must execute only explicit SAN/UCI-like moves after ambiguity is resolved.
   - underspecified natural-language move requests need resolver/clarification behavior.

3. Tightened evaluator:
   - added final-answer claim taxonomy
   - added evaluator conformance against one-bug-per-trace mutants
   - required bad traces for wrong tool, wrong args, wrong state mutation, invented result, ungrounded final claim, prompt injection, and leakage violation

4. Expanded scenario packs:
   - underspecified natural-language move
   - castling rights
   - en passant
   - promotion
   - check/checkmate/stalemate/draw
   - long-game state drift

5. Tightened prompt-injection handling:
   - user-controlled strings in tool results are data, not instructions
   - narrator should receive structured/escaped fields, not raw instruction-like text

6. Updated synthesis:
   - adversarial audit findings merged
   - new consensus items added
   - gaps and recommendations expanded

## New report

- `findings/adversarial_pipeline_audit.md`

## Remaining attack surface for next loop

1. Contract bundle still needs concrete schema skeletons.
2. Evaluator mutation suite needs explicit mutant table.
3. Scenario packs need concrete seed list and coverage matrix.
4. Baseline report needs target model matrix and measurement protocol.
5. Release manifest needs exact required fields and pass/fail semantics.
6. Education/no-position path still needs clearer choice: direct answer, KB tool, or out of MVP.
7. Move resolver design may need separate tool vs router policy decision.
