# LM Router Validation On A Drive Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete real local Qwen/Gemma router validation using only assets stored on disk `A:` and refresh evaluation artifacts with honest blockers.

**Architecture:** Keep existing local-only LM SFT path in `product_demo/train_sft_poc.py` and existing eval path in `product_demo/evaluate_demo.py`. Add only minimal glue needed to support A-drive-local weights/dependencies workflow, then run real training/eval commands to produce evidence-backed artifacts and stop-hook-safe completion state.

**Tech Stack:** Python, torch, transformers, accelerate, python-chess, local filesystem on `A:`

---

## File Structure

- `product_demo/train_sft_poc.py`
  - Existing LM training entrypoint. Keep as single source for local Qwen/Gemma SFT, readiness gating, and output artifact generation.
- `product_demo/evaluate_demo.py`
  - Existing end-to-end LM/router and engine eval entrypoint. Use unchanged unless real run exposes a concrete bug.
- `product_demo/README.md`
  - Update only if real A-drive local-weights workflow needs explicit run instructions.
- `product_demo/poc_models/`
  - Output location for trained local LM artifacts and `sft_eval.json`.
- `results/production_eval/`
  - Output location for refreshed end-to-end eval artifacts.
- `A:/...` local model/dependency storage
  - Source of downloaded weights and optional wheel/cache assets. Must remain on A drive only.

---

### Task 1: Verify A-drive-only local model/dependency workflow

**Files:**
- Modify: `product_demo/README.md` only if workflow differs from current documented commands
- Test: none

- [ ] **Step 1: Inspect current A-drive-only constraints and current LM command surface**

Read these exact areas before changing anything:

```python
# product_demo/train_sft_poc.py
parser.add_argument("--trainer", choices=["linear", "qwen", "gemma4"], default="linear")
parser.add_argument("--model-path")
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--grad-accum-steps", type=int, default=1)
```

And verify current blocker logic:

```python
is_lm_trainer = trainer in {"qwen", "gemma4"}
if is_lm_trainer and not training.get("local_transformers_sft_completed"):
    blockers.append("local_transformers_sft_not_completed")
if is_lm_trainer and not training.get("batching_enabled"):
    blockers.append("lm_batching_not_enabled")
```

- [ ] **Step 2: Verify current documented run commands match A-drive-only requirement**

Run:

```bash
python -m py_compile "A:\Download\llm_tool_calling_research_workspace\product_demo\train_sft_poc.py"
```

Expected: no output

Then inspect whether `product_demo/README.md` uses placeholder model paths instead of explicit A-drive-only examples.

- [ ] **Step 3: Update README only if it lacks explicit A-drive-only LM examples**

If needed, add commands like:

```bash
python product_demo/train_sft_poc.py --train product_demo/training_data/train_sft.jsonl --eval product_demo/training_data/eval_sft.jsonl --out-dir product_demo/poc_models --device cuda --epochs 5 --trainer qwen --model-path A:/models/qwen-small --batch-size 1 --grad-accum-steps 8 --manifest product_demo/training_data/manifest.json --stockfish-path A:/tools/stockfish/stockfish.exe
```

And:

```bash
python product_demo/evaluate_demo.py --model product_demo/poc_models/router_model.json --engine-model product_demo/poc_models/chess_engine_model.json --eval product_demo/training_data/eval_sft.jsonl --out-dir results/production_eval --device cuda --stockfish-path A:/tools/stockfish/stockfish.exe
```

- [ ] **Step 4: Re-read README snippet after edit for exact path consistency**

Expected: all local-model examples use `A:/...` paths only, no hidden remote/download wording.

- [ ] **Step 5: Commit**

```bash
git add product_demo/README.md
git commit -m "docs: clarify A-drive LM workflow"
```

Skip commit if README unchanged.

### Task 2: Install dependencies and stage local weights on A drive

**Files:**
- Create: none
- Modify: none unless a concrete path/config bug is discovered
- Test: runtime import checks only

- [ ] **Step 1: Create or confirm A-drive locations for local assets**

Verify or create these directories:

```text
A:/models/
A:/wheels/
A:/hf-cache/
```

If weights already exist, note exact path such as:

```text
A:/models/qwen-small/
A:/models/gemma-small/
```

- [ ] **Step 2: Install runtime dependencies without moving assets off A drive**

If online install is allowed:

```bash
python -m pip install transformers accelerate
```

If wheel-only/offline install is required and wheels are on A drive:

```bash
python -m pip install --no-index --find-links "A:/wheels" transformers accelerate
```

Expected: install completes with `Successfully installed` lines.

- [ ] **Step 3: Verify imports explicitly**

Run:

```bash
python -c "import transformers, accelerate; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Acquire small local Qwen or Gemma weights onto A drive**

Acceptable end state:

```text
A:/models/<model-name>/config.json
A:/models/<model-name>/tokenizer.json or tokenizer.model
A:/models/<model-name>/model.safetensors or sharded safetensors
```

Reject paths not on A drive.

- [ ] **Step 5: Verify model directory looks like a local Transformers checkpoint**

Run a check like:

```bash
python -c "from pathlib import Path; p=Path('A:/models/qwen-small'); req=['config.json']; print(all((p/x).exists() for x in req), p.exists())"
```

Expected: `True True`

- [ ] **Step 6: Commit**

```bash
git status
```

Expected: no repo source changes from dependency install alone. No commit required unless tracked docs/config changed.

### Task 3: Run real local LM SFT training

**Files:**
- Modify: `product_demo/train_sft_poc.py` only if real run exposes a concrete bug
- Test: `product_demo/poc_models/sft_eval.json`

- [ ] **Step 1: Start with conservative batch settings for small local LM**

Use command template:

```bash
python "A:\Download\llm_tool_calling_research_workspace\product_demo\train_sft_poc.py" --train "A:\Download\llm_tool_calling_research_workspace\product_demo\training_data\train_sft.jsonl" --eval "A:\Download\llm_tool_calling_research_workspace\product_demo\training_data\eval_sft.jsonl" --out-dir "A:\Download\llm_tool_calling_research_workspace\product_demo\poc_models" --trainer qwen --model-path "A:/models/qwen-small" --device cuda --epochs 5 --batch-size 1 --grad-accum-steps 8 --max-length 256 --max-new-tokens 96 --manifest "A:\Download\llm_tool_calling_research_workspace\product_demo\training_data\manifest.json"
```

If CUDA unavailable or OOM, rerun with:

```bash
--device cpu --batch-size 1 --grad-accum-steps 1
```

- [ ] **Step 2: Verify training run reaches completion, not blocker exit**

Expected: JSON summary printed with fields like:

```json
{
  "trainer": "qwen",
  "production_ready": false,
  "router_end_to_end_accuracy": 0.0
}
```

Any `RuntimeError: qwen trainer blocked:` means workflow still incomplete.

- [ ] **Step 3: Inspect generated LM artifact directory**

Expected files under:

```text
A:\Download\llm_tool_calling_research_workspace\product_demo\poc_models\router_lm_model\
```

Look for:

```text
config.json
model.safetensors or model-*.safetensors
tokenizer.json / tokenizer_config.json
```

- [ ] **Step 4: Inspect generated SFT eval payload for honest LM metadata**

Check these exact keys exist:

```json
training.batch_size
training.grad_accum_steps
training.target_length_summary
training.readiness_scope
training.readiness_limits
readiness.blockers
router.end_to_end_accuracy
```

Expected: `training.qwen_gemma_status == "trained_local_router_sft"`

- [ ] **Step 5: Fix only concrete runtime bugs exposed by real LM run**

Allowed fixes are surgical. Examples:

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

or batch-shape/device fixes inside:

```python
def lm_batch(items: list[LmFeatures]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([item.input_ids for item in items]),
        "attention_mask": torch.stack([item.attention_mask for item in items]),
        "labels": torch.stack([item.labels for item in items]),
    }
```

Do not refactor unrelated files.

- [ ] **Step 6: Re-run exact failing train command after any fix**

Expected: previous concrete error disappears.

- [ ] **Step 7: Commit**

```bash
git add product_demo/train_sft_poc.py product_demo/poc_models/sft_eval.json product_demo/poc_models/router_model.json product_demo/poc_models/narrator_model.json
git commit -m "feat: run local qwen router sft"
```

If Gemma used instead, update commit message accordingly.

### Task 4: Run end-to-end LM evaluation and refresh production artifacts

**Files:**
- Modify: `product_demo/evaluate_demo.py` only if real LM eval exposes a concrete bug
- Test: `results/production_eval/*`

- [ ] **Step 1: Run production eval against freshly trained local LM artifact**

Run:

```bash
python "A:\Download\llm_tool_calling_research_workspace\product_demo\evaluate_demo.py" --model "A:\Download\llm_tool_calling_research_workspace\product_demo\poc_models\router_model.json" --engine-model "A:\Download\llm_tool_calling_research_workspace\product_demo\poc_models\chess_engine_model.json" --eval "A:\Download\llm_tool_calling_research_workspace\product_demo\training_data\eval_sft.jsonl" --out-dir "A:\Download\llm_tool_calling_research_workspace\results\production_eval" --device cuda
```

Fallback if needed:

```bash
--device cpu
```

- [ ] **Step 2: Verify eval completes and writes refreshed artifacts**

Expected files:

```text
results/production_eval/sft_prompt_simulation.json
results/production_eval/engine_match_results.json
results/production_eval/engine_backend.json
results/production_eval/summary.md
```

- [ ] **Step 3: Inspect LM eval for real end-to-end fields**

Check output/report for:

```json
router_tool_accuracy
router_end_to_end_accuracy
tool_success_rate
zero_ply_games
```

And inside prompt simulation rows:

```json
raw_generation
parse_ok
parse_error
predicted_arguments
argument_ok
```

- [ ] **Step 4: Fix only concrete eval bugs exposed by real LM artifact**

Keep fixes small. Relevant code areas:

```python
lm_predictor = RouterLmPredictor(model, device) if is_lm else None
predicted, predicted_arguments, raw_generation, parse_error = predict_router_lm(model, case["prompt"], predictor=lm_predictor)
```

And:

```python
end_to_end_correct = sum(1 for row in rows if row.get("parse_ok", True) and row["router_ok"] and row["argument_ok"] is not False)
```

- [ ] **Step 5: Re-run exact eval command after any fix**

Expected: eval finishes and artifact fields remain honest.

- [ ] **Step 6: Commit**

```bash
git add product_demo/evaluate_demo.py results/production_eval/sft_prompt_simulation.json results/production_eval/engine_match_results.json results/production_eval/engine_backend.json results/production_eval/summary.md
git commit -m "test: refresh local LM production eval"
```

Skip code file in commit if unchanged.

### Task 5: Final verification for stop-hook-safe completion

**Files:**
- Modify: `product_demo/write_poc_results.py` or `product_demo/README.md` only if final artifact wording is inconsistent with real LM state
- Test: final train/eval commands plus artifact inspection

- [ ] **Step 1: Re-run syntax verification for touched Python files**

Run:

```bash
python -m py_compile "A:\Download\llm_tool_calling_research_workspace\product_demo\train_sft_poc.py" "A:\Download\llm_tool_calling_research_workspace\product_demo\evaluate_demo.py" "A:\Download\llm_tool_calling_research_workspace\product_demo\write_poc_results.py"
```

Expected: no output

- [ ] **Step 2: Verify linear run no longer has LM-specific blocker leakage**

Run:

```bash
python "A:\Download\llm_tool_calling_research_workspace\product_demo\train_sft_poc.py" --train "A:\Download\llm_tool_calling_research_workspace\product_demo\training_data\train_sft.jsonl" --eval "A:\Download\llm_tool_calling_research_workspace\product_demo\training_data\eval_sft.jsonl" --out-dir "A:\Download\llm_tool_calling_research_workspace\product_demo\poc_models" --trainer linear --device cpu --epochs 5
```

Expected blockers do **not** include:

```text
lm_batching_not_enabled
llm_training_status_not_complete
```

- [ ] **Step 3: Verify LM run now exists as transcript evidence**

Confirm these are true from current artifacts/logs:

```text
training.trainer == qwen or gemma4
training.qwen_gemma_status == trained_local_router_sft
training.local_transformers_sft_completed == true
```

- [ ] **Step 4: Verify final blockers are honest, not integration bugs**

Acceptable remaining blockers:

```text
router_eval_minimum_support_not_met
router_end_to_end_accuracy_below_0.95
real_kaggle_manifest_not_production_valid
stockfish_not_available_for_calibration
```

Unacceptable remaining blockers after successful LM run:

```text
missing_local_model_path
missing_python_package:transformers
missing_python_package:accelerate
local_transformers_sft_not_completed
llm_training_status_not_complete
```

- [ ] **Step 5: Update wording only if results files contradict real LM state**

If needed, align wording in result writers with exact real state, for example:

```python
f"Trained router path uses {training.get('trainer', 'unknown')} on {training.get('device', 'unknown')}; Qwen/Gemma status is {training.get('qwen_gemma_status', 'not_reported')}."
```

- [ ] **Step 6: Final commit**

```bash
git add product_demo/train_sft_poc.py product_demo/evaluate_demo.py product_demo/write_poc_results.py product_demo/README.md product_demo/poc_models/sft_eval.json results/production_eval/sft_prompt_simulation.json results/production_eval/engine_match_results.json results/production_eval/engine_backend.json results/production_eval/summary.md
git commit -m "fix: complete local LM validation"
```

---

## Self-Review

- Spec coverage: plan covers A-drive-only dependency/model staging, real local LM training, end-to-end eval, artifact refresh, blocker validation, and final stop-hook-safe verification.
- Placeholder scan: no TBD/TODO placeholders left; all code-touching steps include exact files/commands/snippets.
- Type consistency: uses existing keys and function names from current codebase: `train_router_lm`, `RouterLmPredictor`, `production_readiness`, `qwen_gemma_status`, `batch_size`, `grad_accum_steps`, `target_length_summary`.
