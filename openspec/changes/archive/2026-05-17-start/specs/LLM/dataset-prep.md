# Dataset Preparation Spec (Phase 1)

## Status
Draft

## Objective
Prepare high-quality SFT dataset aligned to `chess_assistant_sft_dataset_spec_v3.md` turn behavior.

## Source Contract
Reference behavior from:
- `chess_assistant_sft_dataset_spec_v3.md`

Required per-turn behavior:
1. User turn: assistant emits exactly one tool call OR direct reply.
2. Tool-result turn: assistant narrates in human language; no tool call.
3. Assistant never invents tool results.

## Dataset Shape
- JSONL records
- One conversation per record
- Required fields: `id`, `slice`, `messages`, `validated`, `notes`
- Role order valid (`system -> ... user/assistant/tool ...`)

## Slice Quotas
Use v3 distribution unless superseded by explicit signed-off revision.

- A: Move execution (clean)
- B: Move ambiguity loop
- C: Move illegal/invalid
- D: Implicit eval
- E: Best move/continuation
- F: Move-quality review
- G: Threats
- H: Utility tools
- I: Chess knowledge
- J: Plain chat/no tool
- K: Adversarial routing negatives

## Validation Gates (mandatory)

### V1 Schema gate
- Required fields present.
- Tool-call grammar valid for all tool-calling assistant turns.
- Role sequence legal.

### V2 Mode-discipline gate
- Any assistant turn after `tool` role must contain zero tool calls.
- Any assistant tool-calling turn must contain no narration payload.

### V3 Replay gate
Replay each tool call against backend:
- Exact-match family: `move`, `undo`, `legal_moves`, `list_pieces`
- Tolerance family: `eval`, `best_move`, `review_move`, `threats` (per v3 tolerance)
- `ask_chessbot`: replay skip if non-deterministic

### V4 Routing sanity gate
- Slices J, K: zero tool calls.
- Slices A–I: at least one expected tool-family call.

## Batch Admission Policy
- Batch size: 25 conversations per generation call.
- Admit only records passing V1–V4.
- Failed records are discarded and regenerated; no manual patching inside record.

## Data Hygiene
- Deduplicate near-duplicates within slice.
- Keep phrasing diversity high across openers and clarification loops.
- Preserve error cases (timeout/engine_unavailable) at controlled ratio.

## Deliverables
- `data/sft/chess_assistant_v3_train.jsonl`
- `data/sft/chess_assistant_v3_val.jsonl`
- Validation report with pass/fail counts by gate and slice.
