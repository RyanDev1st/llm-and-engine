## 1. Engine capability spec and contracts

- [x] 1.1 Finalize `chess-engine` capability spec requirements and scenarios
- [x] 1.2 Finalize `LLM` modified requirements for engine error/result grounding
- [x] 1.3 Validate OpenSpec artifact structure and delta parsing

## 2. Runtime implementation in src

- [x] 2.1 Implement per-conversation engine session lifecycle manager
- [x] 2.2 Implement deterministic schemas for `move`, `undo`, `legal_moves`, `list_pieces`, `eval`, `best_move`, `review_move`, `threats`
- [x] 2.3 Implement canonical error normalization (`timeout`, `engine_unavailable`, `invalid_position`, `invalid_move`)
- [x] 2.4 Wire engine adapter into tool backend used by replay validator

## 3. Validation and quality gates

- [x] 3.1 Extend replay validator with exact/tolerance policy updates for engine tools
- [x] 3.2 Add red-team probes for engine error classes and state corruption traps
- [x] 3.3 Run gates and produce audit outputs for engine phase
- [x] 3.4 Patch failing buckets by targeted regeneration until freeze condition passes

## 4. OpenSpec completion

- [x] 4.1 Mark all tasks complete with evidence references
- [x] 4.2 Archive change after specs and gates are green
