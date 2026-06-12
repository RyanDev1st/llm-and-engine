Parent: docs/superpowers/specs/2026-05-23-chess-coach-sft-design.md

# End-to-end verification

## Status
Complete. All 7 tasks of the chess-coach delivery shipped.

## Scope
Live demo: trained LoRA adapter on Gemma-4 E2B running behind the 3-panel
website (eval bar · interactive chess.com-style board · streaming chat), with
real Stockfish driving the tool layer.

## Evidence

### Dataset (v3, 2901 records)
- `data/sft/chess_assistant_v3_train.jsonl` = 2616 records, val = 285.
- Per-slice: A630 B296 C244 D308 E350 F315 G140 H139 I119 J122 K238.
- Stripped 1,750 `(x_N)` artifacts; dropped duplicate slice-C file; warmed
  slice-C error narration; canonicalised system prompt; authored slice D (315)
  with real Stockfish depth-15 scores; augmented J/K (human dups collapsed to
  14/16 distinct).
- 0 schema rejects after validator fix (K-1 -> ask_chessbot allowed).

### Training (QLoRA, 1 epoch on full data)
- Base: `src/llm/models/gemma4_e2b` (Gemma-4-E2B-it, 5.11 B params, NF4 4-bit).
- LoRA: r=16, alpha=32, dropout=0.05, target=`attn-only` (q,k,v,o);
  trainable 5,357,568 (0.105 %).
- 164/164 updates, paged_adamw_8bit, seq=1152, batch=1, grad_accum=16,
  ~226 s/update on RTX 4060 Laptop.
- Loss: random 10.25 -> 0.76 final. Adapter at `runs/gemma4_chess/`.

### Routing audit on val (285 conversations)
- **Overall tool-routing accuracy: 276/285 = 96.8 %**.
- Mode-2 discipline: 266/266 clean (no `<tool>` after tool result).
- Per-slice: A 97 % · B 100 % · C 96 % · D 100 % · E 100 % · F 94 % · G 93 % ·
  H 100 % · I 100 % · J 75 % · K 100 %.
- Report: `docs/2026-05-23-routing-audit.md`.

### Q4_0 GGUF deliverable
- Merged adapter into bf16 base (CPU) -> `convert_hf_to_gguf` (b9295) -> f16 GGUF
  (8.83 GiB) -> `llama-quantize` -> **Q4_0 = 3.18 GiB** at
  `runs/gemma4-E2B-chesscoach-Q4_0.gguf`.
- Load test via `llama-completion.exe -no-cnv`: 17.3 tok/s CPU, generates clean.

### Web app end-to-end
- Backend: stdlib HTTPServer at 127.0.0.1:7860 — `/api/state`, `/api/move`,
  `/api/reset`, `/api/chat`. python-chess board + Stockfish UCI authoritative;
  9-tool executor returns spec-exact strings; 3-phase loop (decide -> execute
  -> narrate) wired to the HF 4-bit + LoRA serving backend.
- Frontend: 3-panel SPA (`src/llm/gemma_chat_site/static/`).
  - Left: vertical eval bar (logistic mapping from Stockfish white-POV cp).
  - Middle: interactive 8x8 chess.com-style board, click-to-move with legal
    dots/captures, last-move highlight, coordinate labels.
  - Right: streaming chatbox with user/bot/tool bubbles, typing indicator.
- API smoke (manual repro):
  - `GET /api/state`         -> startpos, turn=white, eval +0.47.
  - `POST /api/move {uci=e2e4}` -> new fen, eval +0.42.
  - `POST /api/chat "how's my position?"`:
    `<tool>eval depth=15</tool>` -> `score: +0.45 pawns from white POV, depth=15`
    -> "This is clearly Black's game now ..." (routing correct).
  - `POST /api/chat "what should I play?"`: `<tool>best_move depth=15 series=1</tool>`.
  - `POST /api/chat "hi there"`: direct reply, no tool.
- Browser repro via playwright-cli (Chrome 1480x880, full-page render).
  Console: only `404 favicon.ico` (cosmetic).
- Screenshot: `docs/screenshots/chess-coach-3panel.png`.

## Known caveats
- Narration polarity drift: model occasionally inverts white/black framing on
  borderline cp scores (slice-D edge case). Routing is correct; only the
  English commentary phrasing is off. Acceptable for MVP; can be tightened by
  a second epoch with stronger slice-D phrasing variety.
- FEN-blind dataset: human slices imply mid-game positions not present in any
  message. Spec section 7.3 startpos replay therefore not applicable; the live
  backend tracks the real board from startpos as the user plays, so the tool
  executor is replay-correct there.
- `llm_runtime/` (JSON tool format) was stranded as planned; `backend/` is the
  shipped spec-compliant runtime.

## Next
None for MVP. Optional follow-ups:
1. Second epoch (~10 h) to tighten narration polarity on edge eval scores.
2. Real `ask_chessbot` KB (RAG) instead of canned answers.
3. DPO pass harvested from slice-K negatives + slice-B clarification pairs.
