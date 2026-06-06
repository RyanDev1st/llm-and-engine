# Handoff: two-path remote model plan

## Current strategy

We now have two active paths:

1. **Primary path: FPT H100 + Qwen**
   - Use FPT subscription/free credit first.
   - Target model: `qwen3.6:27b-q4_K_M` through Ollama.
   - Target hardware: FPT H100, preferably VM; AI Notebook acceptable if VM capacity is unavailable.
   - Source of truth: `implementation_fpt.md`.

2. **Fallback path: Kaggle T4 x2 + smaller Gemma**
   - Use only if FPT path fails or wastes time.
   - Target model: existing Gemma training path.
   - Target hardware: Kaggle Notebook with dual T4.
   - Source of truth: `implementation.md`.

## Decision rule

Try FPT first because subscription credit should be used before switching platforms. If FPT works, continue Qwen path. If FPT fails due to capacity, runtime instability, missing GPU, Ollama/Qwen pull issues, notebook state loss, or backend startup blockers, switch to Kaggle and train smaller Gemma.

## FPT path success criteria

FPT path is accepted only if all checks pass:

- H100 visible through `nvidia-smi` or `torch.cuda.is_available()`.
- Ollama installs and serves without `$HOME is not defined` panic.
- `qwen3.6:27b-q4_K_M` pulls successfully.
- Repo source is available in runtime.
- Backend starts with Ollama model.
- `/api/state` returns JSON.
- Latency is usable enough for chess-coach testing.
- No secrets/tokens printed.

## FPT path failure triggers

Switch to Kaggle if any of these persist after one focused fix cycle:

- no GPU or wrong runtime,
- FPT VM/Notebook capacity unavailable,
- Qwen pull fails or restarts repeatedly,
- Ollama cannot run reliably,
- notebook kernel state keeps breaking cells,
- backend cannot start,
- time/cost burn exceeds likely value.

## Kaggle fallback command

First bounded Gemma run:

```bash
python -m llm_training.run_train \
  --max-steps 500 \
  --rank 4 \
  --targets qv \
  --grad-accum 1 \
  --output gemma4_chess_kaggle_t4x2
```

Artifact export after training:

```bash
cd /kaggle/working/llm-and-engine
zip -r /kaggle/working/gemma4_chess_kaggle_t4x2.zip runs/gemma4_chess_kaggle_t4x2
```

## Current repo facts

- `implementation_fpt.md` now defines FPT/Qwen-first plan.
- `implementation.md` defines v1.2 SFT + Kaggle/Gemma fallback plan.
- `src/llm/backend/model_ollama.py` supports Ollama model backend.
- `src/llm/backend/server.py` can use Ollama fallback if GGUF path missing.
- `src/llm/llm_training/fpt_qwen_scout_v2.ipynb` contains FPT scout cells with `HOME` and branch fallback fixes.
- `scripts/fpt_qwen_vm_bootstrap.sh` exists for VM path if capacity opens.
- `src/llm/llm_training/run_train.py` supports bounded non-smoke training.
- `src/llm/llm_training/test_training_defaults.py` covers bounded training config.

## Immediate next action

Continue FPT H100 attempt first. Run FPT preflight, Ollama startup, Qwen pull, repo setup, backend smoke. If FPT hard-fails, stop and build Kaggle Gemma notebook.

## Constraints

- Never touch `legacy/`.
- Do not commit secrets, tokens, `.env`, Kaggle credentials, Hugging Face tokens, or model weights.
- Stage intended files only; do not use `git add -A`.
- Watch long-running shell/cell output.
- Strategically clean GPU memory when needed.
- Do not confuse FPT/Qwen hosting success with v1.2 SFT training success.

## Pending tasks

- Launch real v1.2 training: now depends on path decision.
  - FPT success: continue Qwen path and decide if training/eval happens there.
  - FPT failure: Kaggle Gemma bounded training.
- Run post-training routing audit after a real artifact exists.
