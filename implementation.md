# Chess-Coach Agent: Kaggle-Train / Local-Host Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a Gemma-4 chess-coach **agent** (tool-router/narrator) as a LoRA adapter — trained via QLoRA on Kaggle T4, served locally as q4_0 GGUF on an RTX 4060 (8GB).

**Architecture:** The agent never computes chess; it routes user intent to the backend's tools and narrates results (`src/llm/llm_training/system_prompt.py`). We fix the v1.2 SFT generator (the current corpus fabricates tool results), regenerate, train E4B QLoRA on Kaggle T4 (16GB — E4B training won't fit the local 8GB), then merge + export q4_0 GGUF for local serving (E4B inference ≈4.5GB, fits).

**Tech stack:** Python, `transformers`, `peft`, `bitsandbytes`, `trl`, `datasets`, `python-chess`, Stockfish, llama.cpp (`src/llm/runtime/llamacpp`), Kaggle Notebook (T4), RTX 4060.

**Model decision:** Gemma 4 **E4B** preferred (`google/gemma-4-E4B-it-qat-q4_0-unquantized` for training, `…-q4_0-gguf` for serving). Fall back to **E2B** only if E4B QLoRA OOMs on T4 or local E4B serving is too slow. See memory `chess-agent-train-host-split`.

---

## Three phases

1. **Phase 1 — Fix the data (BLOCKER).** The v1.2 generator hardcodes `move san=e4` + fabricated `success: e4` (59% illegal vs the row's FEN) and templated persona finals (93% val-target leak). No training is worth running until this is fixed at the generator, then regenerated.
2. **Phase 2 — Train on Kaggle T4.** Parametrize the trainer for E4B, build a Kaggle QLoRA notebook, produce a LoRA adapter, run the routing audit.
3. **Phase 3 — Serve locally.** Merge adapter → export q4_0 GGUF → wire the GGUF backend → run the web app on the 4060.

Each phase produces a verifiable artifact and can stop/resume independently.

---

## File structure (what each touched file owns)

| File | Responsibility | Phase |
| --- | --- | --- |
| `src/llm/llm_dataset/v1/renderer/chess.py` | Render harness_chess rows. **Must derive board facts from FEN, emit a legal move, real tool echo.** | 1 |
| `src/llm/llm_dataset/v1/board_facts.py` (new) | Pure helper: from FEN → real `board_state` string, legal SAN list, requested-move legality, real `move` tool echo. | 1 |
| `src/llm/llm_dataset/v1/validate.py` | Add hard gate: reject any `move san=X` not legal in `position_fen`; reject `board_state turn=` mismatch. | 1 |
| `src/llm/llm_dataset/v1/renderer/tone.py` | Flatten persona openers (tone is out of scope for SFT). | 1 |
| `src/llm/llm_dataset/v1/build.py` | Split; add cross-split dedup of identical final targets. | 1 |
| `src/llm/llm_dataset/v1/tests/` | TDD for all of the above. | 1 |
| `src/llm/llm_training/run_train.py` | Parametrize `--model` (E2B/E4B) instead of hardcoded `gemma4_e2b`. | 2 |
| `src/llm/llm_training/kaggle_e4b_qlora.ipynb` (new) | Kaggle T4 notebook: env → data → QLoRA → export adapter. | 2 |
| `src/llm/llm_training/eval_routing.py` | Post-train routing audit (already reads v1_2_val). | 2 |
| `src/llm/llm_training/export_gguf.py` (new) | Merge adapter into base, convert + quantize to q4_0 GGUF via llama.cpp. | 3 |
| `src/llm/backend/model_gguf.py` | Already loads a GGUF path via `CHESS_GGUF_PATH`; point at the merged adapter GGUF. | 3 |

---

## PHASE 1 — Fix the data generator (BLOCKER)

### Task 1: Board-facts helper (legal move + real tool echo from FEN)

**Files:**
- Create: `src/llm/llm_dataset/v1/board_facts.py`
- Test: `src/llm/llm_dataset/v1/tests/test_board_facts.py`

- [ ] **Step 1: Write the failing test**

```python
# src/llm/llm_dataset/v1/tests/test_board_facts.py
import chess
from llm_dataset.v1.board_facts import board_state_line, choose_move, move_echo, legal_sans

WHITE_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
BLACK_TO_MOVE = "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 2 2"

def test_board_state_line_reports_real_turn():
    assert "turn=black" in board_state_line(BLACK_TO_MOVE)
    assert "turn=white" in board_state_line(WHITE_START)

def test_chosen_move_is_always_legal():
    for fen in (WHITE_START, BLACK_TO_MOVE):
        san = choose_move(fen, seed=7)
        chess.Board(fen).parse_san(san)  # raises if illegal

def test_requested_move_used_when_legal_else_fallback():
    # e4 legal at start -> honored
    assert choose_move(WHITE_START, seed=1, requested="e4") == "e4"
    # e4 illegal for black here -> fallback to a legal move, never e4
    out = choose_move(BLACK_TO_MOVE, seed=1, requested="e4")
    chess.Board(BLACK_TO_MOVE).parse_san(out)

def test_move_echo_matches_real_backend_success_string():
    assert move_echo(WHITE_START, "e4") == "success: e4"

def test_legal_sans_nonempty():
    assert "e4" in legal_sans(WHITE_START)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/llm/llm_dataset/v1/tests/test_board_facts.py -q`
Expected: FAIL — `ModuleNotFoundError: board_facts`.

- [ ] **Step 3: Implement the helper**

```python
# src/llm/llm_dataset/v1/board_facts.py
from __future__ import annotations

import chess


def _board(fen: str) -> chess.Board:
    return chess.Board(fen)


def board_state_line(fen: str) -> str:
    b = _board(fen)
    turn = "white" if b.turn == chess.WHITE else "black"
    check = "yes" if b.is_check() else "no"
    return f"board_state: turn={turn}, fen={b.fen()}, check={check}, legal_count={b.legal_moves.count()}"


def legal_sans(fen: str) -> list[str]:
    b = _board(fen)
    return [b.san(m) for m in b.legal_moves]


def choose_move(fen: str, seed: int, requested: str | None = None) -> str:
    """Return a legal SAN. Honor `requested` if legal; else deterministic legal pick."""
    b = _board(fen)
    if requested:
        try:
            b.parse_san(requested)
            return requested
        except ValueError:
            pass
    sans = legal_sans(fen)
    return sans[seed % len(sans)]


def move_echo(fen: str, san: str) -> str:
    """Mirror the real backend (src/llm/backend/game.py): success on legal, error otherwise."""
    b = _board(fen)
    try:
        b.parse_san(san)
    except ValueError:
        return "error: illegal, reason=illegal move"
    return f"success: {san}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/llm/llm_dataset/v1/tests/test_board_facts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm/llm_dataset/v1/board_facts.py src/llm/llm_dataset/v1/tests/test_board_facts.py
git commit -m "feat(dataset): add FEN-grounded board-facts helper for row rendering"
```

### Task 2: Rewrite `renderer/chess.py` to use real board facts

**Files:**
- Modify: `src/llm/llm_dataset/v1/renderer/chess.py`
- Test: `src/llm/llm_dataset/v1/tests/test_renderer_chess.py`

- [ ] **Step 1: Write the failing test**

```python
# append to src/llm/llm_dataset/v1/tests/test_renderer_chess.py
import re, chess
from llm_dataset.v1.renderer.chess import render_chess_row

MOVE = re.compile(r"<tool>\s*move\s+san=([^\s<]+)")

def _played(row):
    for m in row["messages"]:
        if m["role"] == "assistant":
            hit = MOVE.findall(m["content"])
            if hit:
                return hit[-1]
    return None

def test_rendered_move_is_legal_and_echo_consistent(make_chess_scenario, fake_annotator):
    # slice A = "play a move" scenario
    row = render_chess_row(make_chess_scenario("A"), fake_annotator)
    san = _played(row)
    assert san is not None
    chess.Board(row["position_fen"]).parse_san(san)            # legal
    # the tool turn after the move must be the real success echo, not hardcoded e4
    msgs = row["messages"]
    move_idx = next(i for i, m in enumerate(msgs)
                    if m["role"] == "assistant" and "move san=" in m["content"])
    assert msgs[move_idx + 1]["content"] == f"success: {san}"

def test_board_state_turn_matches_fen(make_chess_scenario, fake_annotator):
    row = render_chess_row(make_chess_scenario("A"), fake_annotator)
    fen_turn = "white" if row["position_fen"].split()[1] == "w" else "black"
    bs = next(m["content"] for m in row["messages"]
              if m["role"] == "tool" and m["content"].startswith("board_state:"))
    assert f"turn={fen_turn}" in bs
```

(Add `make_chess_scenario` / `fake_annotator` fixtures to `tests/conftest.py` building a real `Scenario` with a known FEN and an annotator stub returning `best_san` from `python-chess`.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest src/llm/llm_dataset/v1/tests/test_renderer_chess.py -q`
Expected: FAIL — current renderer emits constant `e4` / `success: e4` / `turn=white`.

- [ ] **Step 3: Edit the renderer** — replace hardcoded chess content with `board_facts`:

```python
# in src/llm/llm_dataset/v1/renderer/chess.py
from ..board_facts import board_state_line, choose_move, move_echo, legal_sans

# _board_state_text(annotated) -> use real FEN:
def _board_state_text(annotated):
    if annotated is None:
        return "board_state: turn=white, check=no, legal_count=20"
    return board_state_line(annotated.fen)

# slice A move emission (was san=e4 / success: e4):
if scenario.slice == "A":
    san = choose_move(annotated.fen, scenario.seed, requested=_requested_san(scenario))
    messages.append({"role": "assistant", "content": f"<tool>move san={san}</tool>"})
    messages.append({"role": "tool", "content": move_echo(annotated.fen, san)})
```

Also: in `_user_message`, replace the `{san}` placeholder with a **varied legal** move (sample from `legal_sans(annotated.fen)` by seed) instead of literal `"e4"`, and thread that as `requested` into `choose_move` so the user ask and the executed move agree. Update slice B `legal_moves` echo and the slice-F `review_move`/`move` pair to use `board_facts` too. Final text: state the real move + real score, no persona opener.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest src/llm/llm_dataset/v1/tests/test_renderer_chess.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm/llm_dataset/v1/renderer/chess.py src/llm/llm_dataset/v1/tests/
git commit -m "fix(dataset): ground chess rows in real FEN — legal moves, real tool echo"
```

### Task 3: Validator hard-gate on move legality

**Files:**
- Modify: `src/llm/llm_dataset/v1/validate.py`
- Test: `src/llm/llm_dataset/v1/tests/test_validate.py`

- [ ] **Step 1: Write the failing test**

```python
# append to src/llm/llm_dataset/v1/tests/test_validate.py
from llm_dataset.v1.validate import validate_row

def _row_with_move(fen, san):
    return {
        "id": "t", "slice": "A", "kind": "harness_chess", "intent": "i",
        "plugin_context": {"installed": [], "enabled": [], "marketplace": []},
        "skills_index": [], "selected_skills": ["chess-coach"],
        "tool_manifest": [{"name": "move", "args": {"san": "required"},
                           "plugin": "chess-official", "enabled": True, "applies_when": "always"}],
        "expected_tool_calls": ["move"], "grounding_sources": ["board_state"],
        "messages": [{"role": "user", "content": "play"},
                     {"role": "assistant", "content": f"<tool>move san={san}</tool>"},
                     {"role": "tool", "content": f"success: {san}"},
                     {"role": "assistant", "content": "Done."}],
        "acceptance_rules": ["engine_grounded"],
        "position_fen": fen, "stockfish_truth": {"best_san": "e4", "score_cp": 0, "depth": 12},
    }

def test_illegal_move_is_rejected():
    fen = "rnbqkb1r/ppppppp1/5n1p/8/7N/8/PPPPPPPP/RNBQKBR1 b Qkq - 3 3"  # black to move
    errs = validate_row(_row_with_move(fen, "e4"))                       # illegal for black
    assert any("illegal_move" in e.rule for e in errs)

def test_legal_move_passes_legality_gate():
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    errs = [e for e in validate_row(_row_with_move(fen, "e4")) if "illegal_move" in e.rule]
    assert errs == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest src/llm/llm_dataset/v1/tests/test_validate.py -q`
Expected: FAIL — no legality check exists.

- [ ] **Step 3: Add the gate in `validate.py`** (inside `validate_row`):

```python
import re, chess
_MOVE = re.compile(r"<tool>\s*move\s+san=([^\s<]+)")

def _check_move_legality(row, violations):
    fen = row.get("position_fen")
    if not fen:
        return
    board = chess.Board(fen)
    for m in row.get("messages", []):
        if m["role"] != "assistant":
            continue
        hit = _MOVE.findall(m["content"])
        if not hit:
            continue
        try:
            board.parse_san(hit[-1])
        except ValueError:
            violations.append(Violation("illegal_move", f"{hit[-1]} illegal in {fen}"))
```

Call `_check_move_legality(row, violations)` before returning. (Match the existing Violation/return type in the file.)

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest src/llm/llm_dataset/v1/tests/test_validate.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm/llm_dataset/v1/validate.py src/llm/llm_dataset/v1/tests/test_validate.py
git commit -m "feat(dataset): hard-reject rows whose move is illegal in position_fen"
```

### Task 4: Flatten persona openers (tone out of scope)

**Files:**
- Modify: `src/llm/llm_dataset/v1/renderer/tone.py`
- Test: `src/llm/llm_dataset/v1/tests/test_tone.py`

- [ ] **Step 1: Failing test**

```python
# src/llm/llm_dataset/v1/tests/test_tone.py
from llm_dataset.v1.renderer import tone

BANNED = ("no fluff", "cutting to it", "plain take", "straight read", "sure thing", "happy to help")

def test_openers_have_no_persona_tokens():
    pools = tone.OPENERS_WARM + tone.OPENERS_BLUNT + tone.OPENERS_SOCRATIC
    for opener in pools:
        low = opener.strip().lower()
        assert not any(low.startswith(b) for b in BANNED)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest src/llm/llm_dataset/v1/tests/test_tone.py -q`
Expected: FAIL.

- [ ] **Step 3: Replace the opener pools** in `tone.py` with empty-or-neutral connectors (e.g. `("",)`) so finals are plain, grounded sentences. Keep `pick()` working with a single empty string.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest src/llm/llm_dataset/v1/tests/test_tone.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm/llm_dataset/v1/renderer/tone.py src/llm/llm_dataset/v1/tests/test_tone.py
git commit -m "fix(dataset): drop persona openers — SFT trains tool use, not tone"
```

### Task 5: Cross-split dedup (kill val leakage)

**Files:**
- Modify: `src/llm/llm_dataset/v1/build.py`
- Test: `src/llm/llm_dataset/v1/tests/test_build.py`

- [ ] **Step 1: Failing test** — assert no val final-answer text appears in train:

```python
# append to src/llm/llm_dataset/v1/tests/test_build.py
def _final(row):
    return next(m["content"] for m in reversed(row["messages"]) if m["role"] == "assistant")

def test_no_val_final_leaks_into_train(tmp_path):
    # build a small gold set with duplicate finals, run build(), assert disjoint finals
    ...  # construct gold_dir with rows sharing identical final text across the 10%-split boundary
    from llm_dataset.v1.build import build
    train, val = build(gold_dir=tmp_path/"gold", train_path=tmp_path/"tr.jsonl", val_path=tmp_path/"va.jsonl")
    tr = [json.loads(l) for l in (tmp_path/"tr.jsonl").read_text().splitlines()]
    va = [json.loads(l) for l in (tmp_path/"va.jsonl").read_text().splitlines()]
    tr_finals = {_final(r) for r in tr}
    assert not any(_final(r) in tr_finals for r in va)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest src/llm/llm_dataset/v1/tests/test_build.py -q`
Expected: FAIL (current split lets identical finals land in both).

- [ ] **Step 3: In `build()`**, after bucketing, drop any val row whose final text already exists in train (move it to train or discard). Grounded rows from Tasks 1–4 should already be near-unique; this is the safety net.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest src/llm/llm_dataset/v1/tests/test_build.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm/llm_dataset/v1/build.py src/llm/llm_dataset/v1/tests/test_build.py
git commit -m "fix(dataset): dedup val finals against train to stop eval leakage"
```

### Task 6: Regenerate v1.2 + re-run QC

**Files:** none (run scripts). Requires Stockfish at `DEFAULT_SF` (see `annotator.py`).

- [ ] **Step 1: Full suite green**

Run: `python -m pytest src/llm/llm_dataset/v1/tests -q`
Expected: PASS.

- [ ] **Step 2: Regenerate gold + split**

```bash
python -m llm_dataset.v1.generate --profile v1.2
python -m llm_dataset.v1.build    --profile v1.2
```
Expected: `wrote accepted=… rejected=…` then `wrote train=… val=…`.

- [ ] **Step 3: QC gate (must all pass)** — run the QC script and confirm:
  - illegal played moves == **0**
  - val finals also in train < **1%**
  - banned-opener finals == **0**
  - move diversity > 1 distinct SAN
  - 100% rows still start with `load_skill`

- [ ] **Step 4: Commit the corpus**

```bash
git add data/sft/v1_2_train.jsonl data/sft/v1_2_val.jsonl data/sft/v1_2/accepted.jsonl data/sft/v1_2/rejected.jsonl
git commit -m "data(v1.2): regenerate grounded, de-leaked, tone-flat corpus"
```

---

## PHASE 2 — Train E4B QLoRA on Kaggle T4

### Task 7: Parametrize trainer for E4B/E2B

**Files:**
- Modify: `src/llm/llm_training/run_train.py`
- Test: `src/llm/llm_training/test_training_defaults.py`

- [ ] **Step 1: Failing test**

```python
# append to src/llm/llm_training/test_training_defaults.py
def test_model_flag_selects_e4b():
    import argparse
    from llm_training.run_train import build_config, LLM_DIR
    args = argparse.Namespace(smoke=False, max_steps=500, grad_accum=1, epochs=3,
                              max_seq=1280, rank=8, targets="qv", lr=2e-4,
                              eval_every=50, output="gemma4_chess_kaggle", model="gemma4_e4b")
    cfg = build_config(args)
    assert cfg.model_path == LLM_DIR / "models" / "gemma4_e4b"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest src/llm/llm_training/test_training_defaults.py -q`
Expected: FAIL — `Namespace` has no `model`; MODEL is hardcoded.

- [ ] **Step 3: Add `--model` arg** in `run_train.py`:

```python
# main(): add
ap.add_argument("--model", default="gemma4_e2b")
# build_config(): replace MODEL constant use with
model_path=LLM_DIR / "models" / args.model,
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest src/llm/llm_training/test_training_defaults.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm/llm_training/run_train.py src/llm/llm_training/test_training_defaults.py
git commit -m "feat(train): --model flag to select gemma4_e4b or gemma4_e2b base"
```

### Task 8: Kaggle T4 QLoRA notebook

**Files:**
- Create: `src/llm/llm_training/kaggle_e4b_qlora.ipynb`

- [ ] **Step 1:** Author the notebook cells (each self-contained — Kaggle kernels lose state on re-run):
  1. **GPU preflight:** `nvidia-smi`; assert `torch.cuda.is_available()`, print device + free VRAM. Stop if no GPU.
  2. **Source:** clone repo branch (or attach as Kaggle dataset); set `PYTHONPATH=<repo>/src/llm`; print commit hash.
  3. **Deps:** install only what Kaggle's image lacks (`peft trl bitsandbytes datasets python-chess`); do **not** reinstall torch.
  4. **Base model:** download `google/gemma-4-E4B-it-qat-q4_0-unquantized` into `src/llm/models/gemma4_e4b` (HF token via Kaggle Secrets — never inline).
  5. **Data check:** assert `data/sft/v1_2_train.jsonl` + `v1_2_val.jsonl` exist; print row counts.
  6. **Train:** `python -m llm_training.run_train --model gemma4_e4b --max-steps 800 --rank 8 --targets qv --grad-accum 8 --output gemma4_chess_kaggle` with live progress.
  7. **VRAM probe note:** if step 6 OOMs, rerun with `--model gemma4_e2b` (documented fallback).
  8. **Export adapter:** `zip -r /kaggle/working/gemma4_chess_kaggle_adapter.zip runs/gemma4_chess_kaggle` and surface in the output panel.

- [ ] **Step 2: Verify** the notebook runs top-to-bottom on a Kaggle T4 and produces the adapter zip.
  Expected: `runs/gemma4_chess_kaggle/adapter_model.safetensors` present; zip downloadable.

- [ ] **Step 3: Commit**

```bash
git add src/llm/llm_training/kaggle_e4b_qlora.ipynb
git commit -m "feat(train): Kaggle T4 E4B QLoRA notebook producing a LoRA adapter"
```

### Task 9: Post-train routing audit

**Files:**
- Use: `src/llm/llm_training/eval_routing.py` (reads `data/sft/v1_2_val.jsonl`)

- [ ] **Step 1:** Run the audit against the trained adapter:

Run: `python -m llm_training.eval_routing --adapter runs/gemma4_chess_kaggle` (confirm the script's actual flag; inspect `eval_routing.py` first).
Expected: routing accuracy + tool-arg validity report.

- [ ] **Step 2:** Record results in `docs/<date>-e4b-routing-audit.md` (Status / Scope / Evidence / Next).

- [ ] **Step 3: Commit**

```bash
git add docs/*-e4b-routing-audit.md
git commit -m "docs: E4B post-train routing audit results"
```

---

## PHASE 3 — Serve locally on the RTX 4060

### Task 10: Merge adapter + export q4_0 GGUF

**Files:**
- Create: `src/llm/llm_training/export_gguf.py`

- [ ] **Step 1:** Implement: load base + adapter (`peft`), `merge_and_unload()`, save merged HF model, then call llama.cpp `convert_hf_to_gguf.py` + `llama-quantize` (under `src/llm/runtime/llamacpp`) to produce `runs/gemma4_chess_e4b-Q4_0.gguf`.
- [ ] **Step 2:** Run it on the merged adapter (do this where VRAM/RAM allows — merge can run CPU).
  Expected: `runs/gemma4_chess_e4b-Q4_0.gguf` (~4.5GB) created.
- [ ] **Step 3: Commit** (code only — `*.gguf` is gitignored).

```bash
git add src/llm/llm_training/export_gguf.py
git commit -m "feat(serve): merge LoRA adapter and export q4_0 GGUF for local serving"
```

### Task 11: Serve + smoke the web app on the 4060

**Files:**
- Use: `src/llm/backend/model_gguf.py` (honors `CHESS_GGUF_PATH`), `src/llm/backend/server.py`, `src/llm/gemma_chat_site/`

- [ ] **Step 1:** Point the server at the new GGUF:

```bash
$env:CHESS_GGUF_PATH = "A:/Download/llm_tool_calling_research_workspace/runs/gemma4_chess_e4b-Q4_0.gguf"
python -m backend.server
```
Expected: `model loaded (GGUF …)`, then `open http://127.0.0.1:7860`.

- [ ] **Step 2:** Smoke the loop in the browser: type "play e4" → backend executes via real tools → agent narrates `success: e4`. Try an illegal move → agent relays the real `error: illegal`.
  Expected: tool calls are real, narration matches real tool results, board updates.

- [ ] **Step 3:** Record the manual repro + a screenshot in `docs/<date>-local-serve-verify.md`.

- [ ] **Step 4: Commit**

```bash
git add docs/*-local-serve-verify.md
git commit -m "docs: local E4B GGUF serving smoke verification"
```

---

## Self-review

**Spec coverage:** train E4B on Kaggle T4 (Phase 2, Task 8) ✓ · output LoRA adapter (Task 8 step 8) ✓ · host q4_0 GGUF locally (Phase 3) ✓ · fix v1.2 generator prerequisite (Phase 1) ✓ · E2B fallback (Task 8 step 7, Task 7) ✓ · archive FPT plan (done before writing) ✓.

**Placeholder scan:** Task 8 / Task 5-step-3 describe authoring work (notebook cells, dedup body) in prose rather than full code — these are integration artifacts whose exact code depends on Kaggle image + the existing build() shape; each has explicit acceptance criteria. All code-level tasks (1–4, 7) carry complete code.

**Consistency:** model dir names (`gemma4_e4b`/`gemma4_e2b`), output name (`gemma4_chess_kaggle`), GGUF name (`gemma4_chess_e4b-Q4_0.gguf`), and the `move`/`success:`/`board_state:` tool strings match the real backend (`src/llm/backend/tools.py`, `game.py`) across all tasks.

**Open items to confirm during execution:** exact `Violation` type name in `validate.py`; `eval_routing.py` adapter flag; llama.cpp converter script name in the bundled runtime.
