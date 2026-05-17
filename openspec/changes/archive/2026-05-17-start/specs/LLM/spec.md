## ADDED Requirements

### Requirement: Two-phase deterministic turn execution
System SHALL execute each user turn through deterministic phase gating.

1. Router phase receives user message + history + summary.
2. Router emits exactly one JSON object:
   - tool call object, or
   - direct reply object.
3. If tool call object emitted, engine executes tool and appends tool result.
4. Narrator phase receives updated history + summary + tool result and emits narration reply.

#### Scenario: Router and Narrator phase isolation
- **WHEN** one model phase is invoked
- **THEN** only that phase output schema is allowed
- **AND** Router and Narrator never run in same call

#### Scenario: Direct reply termination
- **WHEN** Router emits direct reply object
- **THEN** no tool execution occurs
- **AND** turn terminates

#### Scenario: Narrator invocation gate
- **WHEN** latest role is tool
- **THEN** Narrator phase may be invoked
- **AND** output must be narration_reply schema

### Requirement: Router output strictness
Router output SHALL match one allowed schema and nothing else.

Allowed forms:
```json
{"type":"tool_call","tool":"<name>","args":{...}}
```
```json
{"type":"direct_reply","text":"<human reply>"}
```

#### Scenario: Mixed output rejection
- **WHEN** Router output contains tool-call and narration payload together
- **THEN** runtime rejects output

#### Scenario: Non-schema JSON rejection
- **WHEN** Router emits JSON not matching allowed schemas
- **THEN** runtime rejects output

#### Scenario: Non-JSON rejection
- **WHEN** Router emits non-JSON payload
- **THEN** runtime rejects output

### Requirement: Narrator output strictness
Narrator output SHALL be narration only and grounded in tool result.

Allowed form:
```json
{"type":"narration_reply","text":"<human-sounding grounded reply>"}
```

#### Scenario: Tool-call leakage rejection
- **WHEN** Narrator output contains tool-call structure
- **THEN** runtime rejects output

#### Scenario: Grounding enforcement
- **WHEN** Narrator emits facts absent from tool result/history
- **THEN** output fails grounding validation

#### Scenario: Human-like tone requirement
- **WHEN** Narrator emits mechanical or template-only tone
- **THEN** output fails tone rubric

### Requirement: Mode-discipline invariants
Role sequence and behavior invariants SHALL hold for runtime and dataset validation.

#### Scenario: Tool-call turn purity
- **WHEN** assistant turn is tool_call
- **THEN** turn contains no narration text

#### Scenario: Post-tool narration purity
- **WHEN** assistant narrates after tool result
- **THEN** turn contains no tool call

#### Scenario: Role order legality
- **WHEN** tool message appears
- **THEN** immediately previous role is assistant tool_call

#### Scenario: Invariant failure reporting
- **WHEN** invariant violation occurs
- **THEN** validator returns hard failure
- **AND** includes failing turn index and invariant id

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

### Requirement: History and summary context contract
Model calls SHALL include both message history and compact summary.

#### Scenario: Router payload completeness
- **WHEN** Router phase called
- **THEN** payload includes history and summary

#### Scenario: Narrator payload completeness
- **WHEN** Narrator phase called
- **THEN** payload includes history, summary, and latest tool result

#### Scenario: Missing summary contract failure
- **WHEN** summary omitted from phase payload
- **THEN** integration contract check fails

## MODIFIED Requirements

None.

## REMOVED Requirements

None.

## RENAMED Requirements

None.
