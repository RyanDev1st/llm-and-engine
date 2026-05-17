# chess-engine Specification

## Purpose
TBD - created by archiving change chess-engine-start. Update Purpose after archive.
## Requirements
### Requirement: Engine session state lifecycle
System SHALL maintain per-conversation chess state across tool calls.

#### Scenario: Session initialization
- **WHEN** first stateful engine tool call arrives for conversation id
- **THEN** system creates new engine session state
- **AND** session state starts from standard initial position unless explicit valid FEN provided

#### Scenario: State continuity
- **WHEN** subsequent stateful tool calls arrive with same conversation id
- **THEN** system uses existing session state
- **AND** returned board state reflects all prior applied moves

#### Scenario: Session reset
- **WHEN** reset command or explicit new-game operation is invoked
- **THEN** system replaces prior session state with clean initial state

### Requirement: Move legality and apply contract
System SHALL validate and apply moves deterministically.

#### Scenario: Legal move application
- **WHEN** tool `move` receives legal move under current position
- **THEN** system applies move
- **AND** returns updated canonical board representation

#### Scenario: Illegal move rejection
- **WHEN** tool `move` receives illegal move under current position
- **THEN** system returns `invalid_move` error code
- **AND** session state remains unchanged

#### Scenario: Undo consistency
- **WHEN** tool `undo` invoked with non-empty move history
- **THEN** system reverts exactly one ply
- **AND** resulting board matches prior known position

### Requirement: Deterministic tool output schemas
Each chess engine tool SHALL return strict JSON schema output.

#### Scenario: Success envelope
- **WHEN** tool execution succeeds
- **THEN** output includes `ok=true`, `tool`, `state`, and tool-specific payload fields

#### Scenario: Error envelope
- **WHEN** tool execution fails
- **THEN** output includes `ok=false`, `error_code`, `message`, and optional `detail`

#### Scenario: Schema enforcement
- **WHEN** output omits required fields or changes required field types
- **THEN** validator marks hard failure

### Requirement: Engine failure normalization
Backend failures SHALL be normalized into canonical error classes.

#### Scenario: Timeout normalization
- **WHEN** backend exceeds configured tool timeout
- **THEN** system returns `timeout` error code

#### Scenario: Availability normalization
- **WHEN** backend process unavailable or startup fails
- **THEN** system returns `engine_unavailable` error code

#### Scenario: Position input failure
- **WHEN** FEN or state payload invalid
- **THEN** system returns `invalid_position` error code

### Requirement: Eval-family tolerance policy
Eval-like tools SHALL follow tolerance-based validation bounds.

#### Scenario: Eval score tolerance
- **WHEN** tool `eval` returns score
- **THEN** replay validator accepts score within configured numeric tolerance band

#### Scenario: Best-move tolerance set
- **WHEN** tool `best_move` returns candidate move
- **THEN** replay validator accepts move if inside configured top-k engine candidates

#### Scenario: Exact-family exclusion
- **WHEN** tool is `move`, `undo`, `legal_moves`, or `list_pieces`
- **THEN** replay validator requires exact-match behavior

