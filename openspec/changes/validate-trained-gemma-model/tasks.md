## 1. Validation Harness

- [x] 1.1 Locate current Gemma training and inference entry points under `legacy/product_demo/`.
- [x] 1.2 Add an artifact-driven validation command that accepts a trained Gemma artifact path.
- [x] 1.3 Add deterministic chess assistant validation prompts covering tool selection, JSON arguments, pre-tool silence, and post-tool narration.
- [x] 1.4 Validate tool-call JSON against existing tool schemas before semantic scoring.

## 2. Tests

- [x] 2.1 Add unit tests for artifact-path failure handling and result classification.
- [x] 2.2 Add tests for valid and invalid tool-call JSON parsing.
- [x] 2.3 Add regression tests for pre-tool narration rejection and post-tool narration acceptance.
- [x] 2.4 Run `python -m pytest legacy/tests/ -q` and fix failures.

## 3. Model Evaluation

- [x] 3.1 Run validation against the available trained Gemma artifact or record the missing-artifact blocker.
- [x] 3.2 Separate hardware/load failures from model-quality failures in the output.
- [x] 3.3 Record metrics for tool name accuracy, JSON-schema validity, turn discipline, and post-tool answer quality.
- [x] 3.4 Compare results against previous trained artifacts when available.

## 4. Evidence and Promotion

- [x] 4.1 Write a validation report under `legacy/findings/` with commands, artifact paths, metrics, failures, and next actions.
- [x] 4.2 Define first baseline promotion thresholds from observed validation results.
- [x] 4.3 Ensure no local model weights, runtime caches, or generated large artifacts are staged.
