## ADDED Requirements

### Requirement: Validate trained Gemma artifacts
The system SHALL validate a named trained Gemma artifact before it is promoted or used as the accepted chess assistant model.

#### Scenario: Artifact validation requested
- **WHEN** an operator provides a trained Gemma artifact path
- **THEN** the validation run MUST load that artifact or fail with a clear artifact-load error

#### Scenario: Missing artifact
- **WHEN** an operator provides an artifact path that does not exist
- **THEN** the validation run MUST fail before any model-quality result is recorded

### Requirement: Check tool-call format
The system SHALL check trained Gemma outputs for valid tool-call JSON and schema-compliant arguments.

#### Scenario: Valid tool call
- **WHEN** a validation prompt requires a chess tool call
- **THEN** the model output MUST contain the expected tool name and JSON arguments that validate against the tool schema

#### Scenario: Invalid tool call
- **WHEN** a model output includes malformed JSON or schema-invalid arguments
- **THEN** the validation run MUST record the case as a tool-call failure with the prompt identifier

### Requirement: Check tool-call turn discipline
The system SHALL detect whether trained Gemma emits tool calls without narration before tool execution and emits narration after tool results.

#### Scenario: Tool call before narration
- **WHEN** a validation prompt requires tool use before answering
- **THEN** the model output MUST emit the tool call without explanatory narration in the same pre-tool turn

#### Scenario: Narration after tool result
- **WHEN** a tool result is provided to the model after a valid tool call
- **THEN** the model response MUST narrate from the tool result without making another unnecessary tool call

### Requirement: Report validation evidence
The system SHALL write a validation report with commands, artifact paths, metrics, failures, and next actions.

#### Scenario: Validation completes
- **WHEN** validation finishes with pass or fail status
- **THEN** the system MUST write an evidence report under `legacy/findings/` using the project report layout

#### Scenario: Validation blocked by hardware
- **WHEN** validation cannot run because local hardware cannot load or execute the model
- **THEN** the system MUST report the hardware blocker separately from model-quality metrics
