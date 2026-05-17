## MODIFIED Requirements

### Requirement: Error interpretation behavior
Narrator SHALL interpret timeout/availability/invalid tool-result states safely and consistently.

#### Scenario: Failure transparency
- **WHEN** tool result indicates failure state
- **THEN** Narrator acknowledges failure clearly

#### Scenario: No fabricated success
- **WHEN** tool execution failed
- **THEN** Narrator does not claim successful execution

#### Scenario: Safe next step
- **WHEN** error class is timeout/unavailable/invalid
- **THEN** Narrator offers safe next step (retry or rephrase)

#### Scenario: Canonical engine error mapping
- **WHEN** tool error_code is `timeout`, `engine_unavailable`, `invalid_position`, or `invalid_move`
- **THEN** Narrator maps response text to that canonical class
- **AND** does not reinterpret error as different class

## ADDED Requirements

### Requirement: Engine result grounding extension
Narrator SHALL ground response on chess-engine payload fields for move/state/eval claims.

#### Scenario: Move claim grounding
- **WHEN** Narrator states move accepted or rejected
- **THEN** claim matches tool payload `ok` and `error_code`

#### Scenario: State claim grounding
- **WHEN** Narrator references board state after tool execution
- **THEN** referenced state matches tool payload state fields

## REMOVED Requirements

None.

## RENAMED Requirements

None.
