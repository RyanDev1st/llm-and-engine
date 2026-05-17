## Why

The trained Gemma path needs evidence that model improvements produce valid chess assistant tool-calling behavior, not only training artifacts. Current validation is split between training attempts, router tests, and manual checks, so regressions can pass without proving trained Gemma emits deterministic tool calls and handles tool results correctly.

## What Changes

- Add a validation capability for trained Gemma model artifacts.
- Define deterministic evaluation scenarios for chess assistant tool selection, JSON arguments, no-narration tool calls, and post-tool narration.
- Require model-improvement work to produce comparable metrics and failure reports before promotion.
- Keep production paths on real local runtimes or documented artifacts; label fixtures explicitly.

## Capabilities

### New Capabilities
- `trained-gemma-validation`: Validates trained Gemma artifacts against chess assistant tool-calling and response-quality requirements.

### Modified Capabilities

## Impact

- Affects `legacy/product_demo/` training and evaluation scripts.
- Affects `legacy/tests/` validation coverage where trained model artifacts or fixtures are testable.
- Affects `legacy/findings/` reports for evidence-backed validation outcomes.
- May touch local model artifact paths under `results/` without committing large generated weights.
