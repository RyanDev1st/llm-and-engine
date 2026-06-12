# Handoff: chess-coach agent — Kaggle-train / local-host

Plan of record: `implementation.md`. This file = current state + what to do next.

## The goal (single path, unchanged)

Train a Gemma-4 chess-coach **agent** (tool-router/narrator, **never** computes chess — the backend does) as a **LoRA adapter** via QLoRA on **Kaggle/Colab T4**, then serve it **locally on the RTX 4060 (8 GB)** as a merged **GGUF** (Q5_K_M) or the HF adapter.

- **Model:** Gemma 4 **E4B** preferred; **E2B** is what currently ships (E4B QLoRA on a single T4 is the open lever — see Open items).
- **Contract:** Option B agentic harness — `load_skill` is a TOOL; skills are context (progressive disclosure: catalog of name+description always in context, body pulled on demand, persists). **ONE tool call per inference step**, many across the loop (act → read → act). Train == serve via one `build_system(skills_index, tool_manifest, plugin_context)` renderer.
- Decision memory: `chess-agent-train-host-split`, `chess-agent-skill-tool-contract`.

## Current state (2026-06-12)

**Training — done, with the critical bug fixed.** E2B QLoRA adapter trained on T4 (Kaggle quota → Colab port `colab_e2b_qlora.ipynb`). The **Gemma-4 chat template silently dropped `role="tool"`**, so the first adapter learned to fabricate tool results. FIXED + retrained: tool-role remap (`bd48f3c0`) + prompt trim (`290b6078`); the retrained adapter grounds correctly (memory `gemma-template-drops-tool-role`). DDP on 2×T4 OOMs E2B → ship single-GPU (`ddp-not-viable-e2b-t4`).

**Serving — live locally.** Model-service split: persistent `backend/model_server.py` holds weights on :7861; the weightless app (`backend/server.py` + `web_app.py`) connects via `CHESS_MODEL_SERVER` and restarts in ~1 s. GGUF default **Q5_K_M** (Q4_0 was 2.4× faster but **fabricated eval numbers** → number-consistency guard + Q5_K_M re-export, `gguf-q4-fabricates-eval`).

**Harness reliability layer (serve-side, no retrain needed).**
- Deterministic routing hints (`backend/tool_hints.py`): user words → explicit tool reminder injected into the system prompt; widened triggers (UCI moves, slang eval, plural/series best-move, puzzle/scramble).
- Coverage guarantee: `matched_calls` maps intent → a REQUIRED tool set; the loop won't finish while one is ungathered (force-routes it). This is what makes multi-tool reliable on a small model. `coverage=False` = ablation.
- Grounding guards: number guard (fabricated eval → real value), move-name guard (fabricated SANs → real engine moves), answer-coverage (append a required fact the reply dropped), skill-announce strip + leadin recovery.
- Game-over short-circuit + generic skill-router (serve-side, `chess-agent-deterministic-routing-hints`).

**"Thinking".** A staged Router+Verifier+Narrator loop was BUILT then **RETIRED** — slower and dumber than the single loop on E2B. Coverage now lives on the proven single loop. The web "🧠 thinking" panel is just live tool-step disclosure (renders only on tool turns); there is no separate thinking model call.

**This session's work (all committed, 118 backend tests green):**
- **True token streaming** — chunked SSE + `nosniff`/2 KB padding (Chromium sniff-buffer) + browser XHR.onprogress (Edge buffers `fetch().getReader()`). Tokens type out live.
- **Plugin bundles** (`backend/plugins/`): chess-official + openings + analysis; registry aggregates enabled bundles' tools+skills into the served manifest. Cross-bundle routing verified live.
- **Eval bar + engine switch + live token meter**: white-POV logistic bar; selectable Stockfish (deep) vs custom **LiquidChess** `StaticEvaluator` from `src/chess_engine` (only the bar switches — analysis tools stay Stockfish); `⚡ generating · N tokens` ticks during streaming.
- **Position tools fixed**: `random_position` (puzzle/scramble/open) now engine-grounds the real best move; **new `fetch_puzzle`** pulls a real rated Lichess puzzle (verified FEN/themes/solution, `/next` FEN derived from PGN, `/daily` fallback, then local bank). Routing for "give me a chess puzzle" / "randomize the fen" fixed; local bank rebuilt (5/8 were mislabeled/dead).
- **Bug fixes**: final reply truncated at 96 tokens → `REPLY_TOKENS=320`; skill-load-ate-the-turn no longer greets instead of answering (`_force_answer` retry); skill/tool announce sentences stripped from the final reply.

**Dev runtime (npm):** `npm run server` = persistent `model_server` (weights, :7861, leave up). `npm run dev` = `dev_serve` weightless app (:7860), auto-restarts on any `backend/*.py` save. Browser **live-reload** polls `/api/dev/reload-token` on localhost → reloads on any frontend/backend save. `npm run dev:solo` = one-process fallback.

## Run commands

```bash
# serve (two terminals)
npm run server            # model_server, weights on :7861 — leave running
npm run dev               # weightless app on :7860, hot-reload; open http://127.0.0.1:7860
#   pass an adapter:  npm run server -- "A:/path/to/adapter"

# tests
python -m pytest src/llm/backend -q                      # 118 serve tests
python -m pytest src/llm/llm_dataset/v1/tests -q          # dataset generator
python -m pytest src/llm/llm_training -q                  # trainer

# data + train + export (see implementation.md)
cd src/llm && python -m llm_dataset.v1.generate --profile v1.2   # then .build
cd src/llm && python -u -m llm_training.run_train                # reads v1_2_train/val
cd src/llm && python -u -m llm_training.export_gguf ../../runs/gemma4_chess
```

## Active corpus

`v1_2` only: `data/sft/v1_2_{train,val}.jsonl(.gz)` + `data/sft/v1_2/{accepted,rejected}`. Stored gzipped (under GitHub 100 MB). Last full regen `8572d7ed`: 0 illegal moves, loaded-skill diversity 447, freeze_ok, 0 undeclared tool calls, 0% val leak. Source of truth for harness behavior = `src/llm/llm_dataset/v1/` (`contracts.py`, `profiles.py`).

## Constraints

- Never edit/import `legacy [ignore]/` (gitignored archive bin).
- No secrets in code/logs/commits; HF token via Kaggle/Colab Secrets.
- Commit every turn; **push/PR only when the user explicitly asks** (branch `feat/chess-coach-sft` — not yet pushed; first push, not a force-push).
- Stage intended files only (no `git add -A`); `*.gguf`/`*.safetensors` gitignored — commit code, not weights.
- Watch long shells; free GPU memory between heavy runs; clean stale :7860/:7861 procs after testing.

## Open items / next

1. **E4B feasibility on a non-FPT cloud** (FPT free A100 unreliable; FPT path abandoned). Single-T4 E4B QLoRA is the untried lever. User is scoping other clouds — deferred until then.
2. **Model ceilings (training lever, not serve):** E2B still fabricates move names, ignores the injected live board, and drops multi-turn intent. The serve guards above mitigate; a stronger base (E4B) or targeted SFT is the real fix. Overlay-following SFT (Option B) deferred until an E4B eval (`chess-agent-prompt-layering`).
3. **Val split small (591):** finals are templated → de-leak shrank val. Fix = split by intent/scenario family in `build.split_train_val` (not by dropping colliding finals).
4. **fen loader:** flagged by the user but no bug reproduced — `load_fen` correctly rejects illegal FENs and the board syncs. Need an exact failing FEN/phrasing to chase.
5. **Push (Phase 4):** on explicit OK, push the branch, run the Kaggle/Colab notebook, export adapter → merge → GGUF → serve smoke on the 4060.
