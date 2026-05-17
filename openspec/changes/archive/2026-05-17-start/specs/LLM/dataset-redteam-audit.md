# Dataset Red-Team + Patch + Audit Spec (Phase 2)

## Status
Draft

## Objective
Stress-test dataset for failure holes, patch systematically, then certify final quality.

## Inputs
- Phase-1 admitted dataset output
- Validation logs (V1–V4)
- Slice-level distribution summary

## Red-Team Scope
Probe for dataset weaknesses that cause bad routing, mode errors, or mechanical narration.

### R-team categories
1. Mode violation traps
   - assistant emits tool call after tool result
   - assistant mixes narration with tool-call turn

2. Ambiguity handling holes
   - options ignored
   - guesses instead of asking clarification

3. Illegal/invalid error handling holes
   - wrong error class narration
   - speculative board claims not present in tool result

4. Tool failure robustness
   - timeout / engine_unavailable mishandled
   - fabricated success after failure payload

5. Adversarial routing negatives
   - off-topic with chess words triggers tool call
   - abstract chess knowledge routed to board tool

6. Injection-style prompts
   - user text attempts to override tool protocol
   - assistant breaks mode discipline

7. Tone quality
   - repetitive mechanical phrasing
   - low variation in equivalent outcomes

## Patch Policy
- No in-place hand edits to failing conversations.
- Patch only by targeted regeneration of failing pattern bucket.
- Track each patch batch with:
  - failure category
  - affected slices
  - replaced record IDs
  - post-patch pass rates

## Audit Protocol

### A1 Coverage audit
- Slice counts meet quota targets.
- Critical slices (B, C, F, K) not underrepresented.

### A2 Invariant audit
- Mode-discipline violations = 0.
- Schema violations = 0.
- Routing sanity violations = 0.

### A3 Replay audit
- Replay pass rate meets target threshold.
- Error-case payloads preserved and correctly narrated.

### A4 Diversity audit
- Near-duplicate rate below threshold.
- Opening phrase diversity across slices.
- Clarification-loop variation present.

### A5 Tone audit
- Narrations judged human-like and non-mechanical on sampled review set.
- No dominant canned template overuse.

## Exit Criteria (dataset freeze)
Dataset can freeze only when all hold:
1. V1/V2/V4 hard-fail counts = 0.
2. Replay gate meets target pass bar.
3. Red-team category pass bars met.
4. Audit report complete and signed off.

## Deliverables
- Red-team findings report (holes + evidence)
- Patch log (before/after metrics)
- Final audit report
- Frozen dataset manifest with checksums
