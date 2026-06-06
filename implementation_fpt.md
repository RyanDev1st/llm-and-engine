# FPT H100 Qwen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the FPT subscription first by running Qwen on FPT H100 for chess-coach inference/training experiments, then fall back to Kaggle T4 x2 with smaller Gemma if FPT fails.

**Architecture:** Keep FPT/Qwen path separate from the original Gemma/Kaggle path. FPT path focuses on using available H100 subscription value for Qwen 27B Q4 hosting and model-assisted iteration; Kaggle path remains fallback for bounded Gemma QLoRA training on smaller hardware.

**Tech Stack:** FPT AI Notebook or VM, H100, Ollama, `qwen3.6:27b-q4_K_M`, Python backend under `src/llm/backend`, existing chess web app, Kaggle T4 x2 fallback, Gemma QLoRA trainer under `src/llm/llm_training`.

---

## Decision

Try FPT first. If FPT H100 path works, use subscription credit and continue Qwen route. If FPT fails due to capacity, notebook limits, Ollama/install friction, model pull failure, runtime instability, or inability to run needed training/eval within time, switch to Kaggle T4 x2 and train smaller Gemma.

## Active paths

### Path A: FPT H100 + Qwen first

Purpose:
- use paid/free FPT subscription credit before spending Kaggle effort,
- validate whether H100 notebook/VM can host Qwen reliably,
- run chess-coach app against Qwen-backed backend,
- optionally run remote training/eval if environment supports it.

Model:
- `qwen3.6:27b-q4_K_M` through Ollama.

Runtime target:
- FPT VM preferred if capacity exists,
- FPT AI Notebook acceptable when VM occupied,
- H100 required for full Qwen path.

Success criteria:
- GPU visible through `nvidia-smi` or `torch.cuda.is_available()`.
- Ollama installs and serves without `$HOME` panic.
- Qwen model pulls successfully.
- Repo source is available in runtime.
- Backend starts and `/api/state` returns JSON.
- Chess web app can hit backend locally or via tunnel.
- No secrets printed.

Failure triggers:
- no GPU available,
- H100 capacity unavailable for meaningful period,
- Ollama cannot run reliably,
- Qwen pull too slow/fails/restarts repeatedly,
- notebook kernel loses state too often,
- app/backend cannot start after bounded fix attempts,
- FPT wall-clock cost outweighs progress.

### Path B: Kaggle T4 x2 + smaller Gemma fallback

Purpose:
- train original Gemma-based v1.2 SFT adapter with conservative settings,
- avoid Qwen/Ollama hosting complexity,
- produce real bounded training artifact and run post-training audit.

Model:
- existing Gemma path from `src/llm/llm_training/run_train.py`.

Runtime target:
- Kaggle Notebook with T4 x2.

First command:

```bash
python -m llm_training.run_train \
  --max-steps 500 \
  --rank 4 \
  --targets qv \
  --grad-accum 1 \
  --output gemma4_chess_kaggle_t4x2
```

Success criteria:
- T4 x2 visible,
- dataset files exist,
- training starts and prints progress,
- adapter output saved,
- artifact exported before Kaggle session ends,
- post-training routing audit runs.

## Files already relevant

- `handoff.md` — current operational handoff; must state two-path strategy.
- `implementation.md` — Gemma/Kaggle fallback plan and v1.2 SFT training plan.
- `implementation_fpt.md` — this FPT/Qwen-first plan.
- `src/llm/backend/model_ollama.py` — Ollama backend for Qwen.
- `src/llm/backend/server.py` — backend model-selection and bind config.
- `src/llm/backend/test_colab_config.py` — env/runtime config tests for backend hosting.
- `src/llm/llm_training/fpt_qwen_scout_v2.ipynb` — FPT notebook scout path.
- `scripts/fpt_qwen_vm_bootstrap.sh` — FPT VM bootstrap path if VM capacity opens.
- `src/llm/llm_training/run_train.py` — Gemma bounded training entrypoint.
- `src/llm/llm_training/test_training_defaults.py` — bounded training config test.

## Task 1: FPT preflight gate

**Files:**
- Use: `src/llm/llm_training/fpt_qwen_scout_v2.ipynb`
- Use: `scripts/fpt_qwen_vm_bootstrap.sh` if VM path opens

- [ ] **Step 1: Start FPT runtime**

Use H100 runtime if available. Prefer VM. Use AI Notebook only if VM remains occupied.

- [ ] **Step 2: Run GPU preflight**

Notebook cell content:

```python
import os
import subprocess

os.environ["HOME"] = os.environ.get("HOME") or "/root"
os.environ["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:" + os.environ.get("PATH", "")

def run(cmd):
    print(">", cmd)
    return subprocess.run(["bash", "-lc", cmd], text=True, check=False, env=os.environ.copy())

run("which nvidia-smi || true")
run("nvidia-smi || true")
run("python - <<'PY'\nimport torch\nprint('cuda', torch.cuda.is_available())\nprint('count', torch.cuda.device_count())\nprint('name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)\nPY")
```

Expected:
- H100 visible or torch CUDA true.

If not visible:
- stop FPT path for this runtime,
- switch runtime or prepare Kaggle fallback.

- [ ] **Step 3: Verify Ollama install/start**

Notebook cell content:

```python
import os
import pathlib
import subprocess
import time

os.environ["HOME"] = os.environ.get("HOME") or "/root"
os.environ["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:" + os.environ.get("PATH", "")
os.environ["OLLAMA_MODELS"] = "/workspace/ollama-models"
pathlib.Path(os.environ["HOME"]).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.environ["OLLAMA_MODELS"]).mkdir(parents=True, exist_ok=True)

def run(cmd, check=False):
    print(">", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=check, text=True, env=os.environ.copy())

run(["bash", "-lc", "DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl zstd"], check=False)
run(["bash", "-lc", "curl -fsSL https://ollama.com/install.sh | sh"], check=False)
subprocess.run(["bash", "-lc", "pkill -f 'ollama serve' || true"], check=False, env=os.environ.copy())
subprocess.Popen(["bash", "-lc", "ollama serve > /tmp/ollama.log 2>&1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=os.environ.copy())
time.sleep(5)
run(["bash", "-lc", "ollama --version"], check=False)
run(["bash", "-lc", "curl -s http://127.0.0.1:11434/api/tags || true"], check=False)
```

Expected:
- no `$HOME is not defined` panic,
- tags endpoint returns JSON.

## Task 2: FPT Qwen model pull

**Files:**
- Use: `src/llm/llm_training/fpt_qwen_scout_v2.ipynb`

- [ ] **Step 1: Pull Qwen only when H100 is visible**

Notebook cell content:

```python
import os
import subprocess

os.environ["HOME"] = os.environ.get("HOME") or "/root"
os.environ["OLLAMA_MODEL"] = "qwen3.6:27b-q4_K_M"
os.environ["OLLAMA_MODELS"] = "/workspace/ollama-models"

def run(cmd):
    print(">", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=True, text=True, env=os.environ.copy())

run(["ollama", "pull", os.environ["OLLAMA_MODEL"]])
```

Expected:
- pull completes.

If pull fails repeatedly:
- record exact error,
- stop FPT path,
- switch to Kaggle fallback.

## Task 3: FPT backend smoke

**Files:**
- Use: `src/llm/backend/model_ollama.py`
- Use: `src/llm/backend/server.py`
- Use: `src/llm/llm_training/fpt_qwen_scout_v2.ipynb`

- [ ] **Step 1: Clone/update repo**

Use `fpt_qwen_scout_v2.ipynb` Cell 5. It falls back to default branch if `feat/chess-coach-sft` is missing upstream.

- [ ] **Step 2: Install runtime deps without reinstalling torch unless needed**

Notebook cell content:

```python
import os
import pathlib
import subprocess

os.environ["HOME"] = os.environ.get("HOME") or "/root"
WORKDIR = pathlib.Path("/workspace/llm-and-engine")
PY = WORKDIR / ".venv" / "bin" / "python"

def run(cmd):
    print(">", " ".join(map(str, cmd)))
    return subprocess.run(cmd, check=True, text=True, env=os.environ.copy())

run(["python3", "-m", "venv", "--system-site-packages", str(WORKDIR / ".venv")])
run([str(PY), "-m", "pip", "install", "--upgrade", "pip"])
run([str(PY), "-m", "pip", "install", "transformers", "accelerate", "peft", "bitsandbytes", "datasets", "sentencepiece", "protobuf", "python-chess"])
run([str(PY), "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"])
```

Expected:
- dependency install succeeds,
- torch import works.

- [ ] **Step 3: Start backend**

Notebook cell content:

```python
import os
import pathlib
import subprocess
import time

WORKDIR = pathlib.Path("/workspace/llm-and-engine")
PY = WORKDIR / ".venv" / "bin" / "python"
os.environ["HOME"] = os.environ.get("HOME") or "/root"
os.environ["OLLAMA_MODEL"] = "qwen3.6:27b-q4_K_M"
os.environ["CHESS_HOST"] = "127.0.0.1"
os.environ["CHESS_PORT"] = "7860"

server_cmd = (
    f"cd {WORKDIR} && "
    f"export HOME={os.environ['HOME']} && "
    f"export PYTHONPATH={WORKDIR / 'src' / 'llm'} && "
    f"export CHESS_HOST={os.environ['CHESS_HOST']} && "
    f"export CHESS_PORT={os.environ['CHESS_PORT']} && "
    f"export OLLAMA_MODEL={os.environ['OLLAMA_MODEL']} && "
    f"{PY} -m backend.server > /tmp/chess_server.log 2>&1"
)
subprocess.Popen(["bash", "-lc", server_cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=os.environ.copy())
time.sleep(3)
print("server starting")
```

- [ ] **Step 4: Check `/api/state`**

Notebook cell content:

```python
import json
import time
import urllib.request

for _ in range(30):
    try:
        body = urllib.request.urlopen("http://127.0.0.1:7860/api/state", timeout=3).read().decode()
        print(json.dumps(json.loads(body), indent=2)[:2000])
        break
    except Exception as exc:
        print("waiting:", exc)
        time.sleep(2)
else:
    print(open("/tmp/chess_server.log", "r", errors="replace").read()[-4000:])
    raise RuntimeError("server did not become ready")
```

Expected:
- JSON response from backend.

## Task 4: FPT accept/reject decision

**Files:**
- Modify if needed: `handoff.md`
- Modify if needed: `implementation_fpt.md`

- [ ] **Step 1: Accept FPT path if all checks pass**

FPT accepted if:
- H100 visible,
- Ollama stable,
- Qwen pulled,
- backend `/api/state` returns JSON,
- inference latency is usable.

- [ ] **Step 2: Reject FPT path if any hard failure persists**

FPT rejected if:
- no H100 visible,
- runtime capacity unavailable,
- model pull impossible,
- notebook state keeps breaking workflow,
- backend cannot start after one focused fix cycle.

- [ ] **Step 3: If rejected, switch to Kaggle plan**

Use `implementation.md` Kaggle Gemma path.

## Task 5: Kaggle fallback

**Files:**
- Use: `implementation.md`
- Use/Create later: `src/llm/llm_training/kaggle_gemma_t4x2_train.ipynb`
- Use: `src/llm/llm_training/run_train.py`

- [ ] **Step 1: Use smaller Gemma path on Kaggle**

First bounded command:

```bash
python -m llm_training.run_train \
  --max-steps 500 \
  --rank 4 \
  --targets qv \
  --grad-accum 1 \
  --output gemma4_chess_kaggle_t4x2
```

- [ ] **Step 2: Export adapter**

After training, zip run output:

```bash
cd /kaggle/working/llm-and-engine
zip -r /kaggle/working/gemma4_chess_kaggle_t4x2.zip runs/gemma4_chess_kaggle_t4x2
```

Expected:
- `/kaggle/working/gemma4_chess_kaggle_t4x2.zip` visible in Kaggle output panel.

- [ ] **Step 3: Run routing audit after artifact exists**

Use existing audit scripts discovered in repo; if command choice is unclear, inspect `scripts/llm_eval.py`, `scripts/llm_validate.py`, and `src/llm/llm_training/eval_routing.py` before running.

## Current source of truth

- FPT/Qwen path: `implementation_fpt.md`.
- Gemma/Kaggle fallback path: `implementation.md`.
- Operational handoff: `handoff.md`.

## Self-review

Spec coverage:
- two-path strategy documented,
- FPT first documented,
- Kaggle fallback documented,
- smaller Gemma fallback documented,
- FPT H100 + Qwen documented.

Placeholder scan:
- no TBD/TODO placeholders.

Type/command consistency:
- Qwen model name consistent: `qwen3.6:27b-q4_K_M`.
- Kaggle output name consistent: `gemma4_chess_kaggle_t4x2`.
