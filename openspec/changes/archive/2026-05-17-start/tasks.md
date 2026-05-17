# Tasks — change `start` (capability: LLM)

## Phase 1 — Dataset preparation

- [x] 1.1 Set canonical tool/turn contract from `chess_assistant_sft_dataset_spec_v3.md`
  - Lock role sequence, tool-call grammar, mode-discipline constraints.
  - Define deterministic acceptance checks for each assistant turn.

- [x] 1.2 Implement generation batch workflow (25 conversations per batch)
  - Slice-targeted generation prompts.
  - Batch metadata: slice, batch_id, generator, timestamp.

- [x] 1.3 Implement schema validator (V1)
  - Required fields check: `id`, `slice`, `messages`, `validated`, `notes`.
  - Tool-call format check.
  - Role-order legality check.

- [x] 1.4 Implement mode-discipline validator (V2)
  - Assistant-after-tool must contain zero tool calls.
  - Assistant tool-call turn must contain no narration payload.

- [x] 1.5 Implement replay validator (V3)
  - Execute tool calls against backend.
  - Exact-match family checks.
  - Tolerance family checks.
  - Non-deterministic `ask_chessbot` policy.

- [x] 1.6 Implement routing sanity validator (V4)
  - Slices J/K enforce zero tool calls.
  - Slices A–I enforce expected tool-family calls.

- [x] 1.7 Build admission pipeline
  - Reject-on-fail for V1–V4.
  - Regenerate failed records only.
  - Emit gate failure reason per rejected record.

- [x] 1.8 Build dataset hygiene pass
  - Near-duplicate detection within slice.
  - Diversity checks for opener phrasing and clarification loops.

- [x] 1.9 Produce phase-1 deliverables
  - `data/sft/chess_assistant_v3_train.jsonl`
  - `data/sft/chess_assistant_v3_val.jsonl`
  - Validation summary report by gate + slice.

## Phase 2 — Red-team, patch, audit

- [x] 2.1 Build red-team probe suite
  - Mode violation traps.
  - Ambiguity handling traps.
  - Illegal/invalid narration traps.
  - Timeout/engine_unavailable traps.
  - Adversarial routing negatives.
  - Injection-style protocol override traps.

- [x] 2.2 Run first red-team pass and log holes
  - Tag each failure by category and slice.
  - Store evidence with failing record IDs.

- [x] 2.3 Patch by targeted regeneration
  - No in-place manual rewrite of failing conversations.
  - Regenerate failing buckets and rerun V1–V4.

- [x] 2.4 Run second red-team pass (regression)
  - Confirm patched categories remain green.
  - Confirm no new regressions in untouched slices.

- [x] 2.5 Perform final dataset audit
  - Coverage audit (slice quota conformance).
  - Invariant audit (V1/V2/V4 hard-fail count = 0).
  - Replay audit (V3 pass threshold met).
  - Diversity audit (duplicate thresholds met).
  - Tone audit (human-like narration sample review).

- [x] 2.6 Freeze dataset artifacts
  - Produce manifest with checksums.
  - Publish red-team report, patch log, final audit report.

## Exit condition

- [x] E1 Dataset freeze approved only if all gates pass:
  - V1/V2/V4 hard-fail = 0
  - V3 pass rate at or above threshold
  - Red-team categories pass
  - Audit report complete and signed off
