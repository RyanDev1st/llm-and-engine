# Presentation report — Gemma 4 E4B chess-coach harness (reading guide)

One place to read before the presentation. It curates the real artifacts, separates **what matters**
from **what to skip**, states every number with its source + honesty caveat, and lists exactly **what
still needs a Kaggle run**. It does not duplicate the findings docs — it points to them.

Last updated 2026-06-25. Model under test: **E4B + v4 LoRA adapter (nf4)**, frozen. All numbers below
are from the **2026-06-24 Kaggle run** unless marked otherwise.

---

## 1. TL;DR — the three numbers that matter + the one-line story

- **Routing (fair test): 88.7% verb accuracy** vs **49.6% for the base model** — the fine-tuning win.
- **Out-of-domain task completion: 91.7% completed, 95% grounded** (60 held-out cooking/music/wellness/
  tax prompts the model never trained on) — it generalizes.
- **The base→adapter recalibration:** tool false-positives **55 → 7**, tool F1 **0.42 → 0.81**, skill F1
  **0.56 → 0.94**.

**Story:** the LoRA fine-tune took a base model that over-fires tools (28% tool precision) and taught it
to route skill-vs-tool correctly and complete unseen-domain tasks, served on a local GGUF runtime.

---

## 2. What to READ (ranked) — and what to SKIP

**Read these three, in order:**
1. **`docs/findings/2026-06-25-train-serve-parity-audit.md`** — the most important finding: the harness
   (not the weights) was capping the chess completion metric. Explains the 13.5% artifact and the fix.
2. **`docs/findings/2026-06-24-harness-live-vs-benchmark-gap.md`** — why "live ≠ benchmark": three
   different prompts; the board injection is off-distribution. Sets up #1.
3. **`docs/findings/2026-06-25-answer-quality-serve-fixes-and-corpus-finding.md`** — the reply-content
   issues (serve fixes landed) + the corpus finding that scopes a future v5 (not done now).

**Skip / background only (do NOT read for the presentation):**
- `2026-06-24-harness-vs-claude-code-codex.md` — design rationale, not results.
- `2026-06-25-serve-latency-investigation.md` — read ONLY if asked "why is it slow" (answer: decode-
  bound, ~2-4 tok/s on T4; GGUF quants are the lever).
- Everything dated 2026-06-06 … 2026-06-22 — superseded by the above.

---

## 3. Measured numbers (every one with its source + honesty caveat)

### 3a. Routing — the FAIR test (use this) · `eval_benchmark --suite val --native`, n=142
| metric | e4b-v4 adapter | e4b base | honest note |
|---|---|---|---|
| verb accuracy | **88.7%** | 49.6% | scored in each row's TRAINED reasoning mode (fair) |
| macro precision | 58.6% | 39.5% | low because `none` has 0 support in this slice mix |
| weighted precision | 95.8% | 78.0% | the practical number |
| exact-name | 55.6% | 13.0% | dragged down by G/H name-slip (see §5) |
| format validity | 91.5% | 81.3% | no foreign tags |
| tool F1 | **0.81** | 0.42 | the recalibration |
| skill F1 | **0.94** | 0.56 | |
| throughput | 12.74 s/row | 14.71 | T4, nf4 |

Adapter confusion (rows=gold): skill `[104,7,6]`, tool `[0,22,3]`, none `[0,0,0]`. The 9 `→none` cells
are the model's one blind spot (answers directly when it shouldn't); minor.

### 3b. Completion — OUT-OF-DOMAIN generalization (use this) · `eval_completion --stress`, n=60
| first_ok | completed | exec_ok | args_ok | grounded | recovered |
|---|---|---|---|---|---|
| 91.7% | **91.7%** | 96.7% | 100% | **95%** | 0% |
Real cooking/music/wellness/tax prompts, never trained on. The strongest single result.

### 3c. Numbers to NOT lead with (misleading without caveat)
- **Confusion run 98.5% verb** (`eval_confusion`, n=196): the chess val slices are **100% skill-gold**
  (the whole val is 2591 skill / 102 tool / 38 none), so this only tests "load the skill," not 3-class
  routing. The report's own macro-precision 33.3% flags it. **Do not present as routing accuracy.**
- **Chess completion 13.5%** (`eval_completion` chess, n=37): a **harness artifact** — the live-board
  injection made `board_state` redundant so the model skipped it and `completed` failed despite correct
  routing. **Fixed, must be re-run at parity (§6).** Do not present the 13.5%.

---

## 4. Images — what each shows + status

| image (Kaggle output) | what it shows | status |
|---|---|---|
| confusion `e4b-v4 adapter` (n=142) | clean skill/tool routing, 88.7% | **REAL — use it** |
| confusion `e4b base` (n=142) | the over-firing baseline (55 skill→tool) | **REAL — use it (the contrast)** |
| confusion `routing-confusion` (n=196, all-skill) | the 98.5% one | REAL but misleading — see §3c, skip or caveat |
| `chat-section1/2` cards + transcript | the agent on OOD domains, with timing/tok-s | **REAL but the showcase ran with a plugin-context BUG** (opening-advisor/tactical-puzzles came back `unknown_skill`); replies are NOT representative — **re-run (§6)** |
| `chart-model-lines` (cross-model) | perf across models | **INCOMPLETE** — only base 83% (seed) + nf4 98% (the misleading confusion number); completion/tok-s + E2B + Q5/Q6 all missing — **re-run (§6)** |

The 3 local PNGs in `docs/findings/report_assets/` (corpus/layer/timeline) are CPU-generated design
charts — fine as background, not results.

---

## 5. Not bugs (so you can answer questions confidently)
- **Slices G (24%) and H (18%):** the model emits `<skill>threats</skill>` — loading a *tool* name as a
  skill — instead of `chess-coach`. Verb is still correct; only exact-NAME misses, and the loop returns
  a corrective and usually **recovers**. A name slip the harness absorbs, not broken routing.
- **The 9 `→none` cells:** minor; do not "fix" by penalizing direct answers (some direct answers are
  correct — `V1_Q` is legitimately none-gold).

---

## 6. PENDING — what to RUN on Kaggle (the gap list)

Pull `feat/report-ppt-assets` (Cell 2 clones it). Priorities for the presentation:

| # | what | how (Kaggle cell / flag) | why it matters |
|---|---|---|---|
| **P1** | **Chess completion AT PARITY** | Cell 6.7 with `RUN_COMPLETION_CHESS=True` (the cell now sets `CHESS_BOARD_HOOK=0`) | replaces the bogus 13.5% with the real number; expected to jump sharply |
| **P2** | **Representative chats + the A/B** | the **CHAT A/B cell** (after Cell 3) with the plugin-context fix | the shown chats were buggy; this gives real replies + board_on/off/thin content + board before→after |
| **P3** | **Feed the cross-model chart** | re-run Cell 6.7 STRESS (now `--tag e4b-nf4`) + the chat cell, then the model-lines cell | adds completion/grounded/tok-s for nf4 so the line chart isn't 2 points |
| P4 | **GGUF Q5_K_M / Q6_K** | finish the Colab export, then Cell 6.8 (`RUN_GGUF_AB=True`) | the speed/quality story (quants ~2.4× faster); optional if time-short |
| P5 | **E2B prior model** | Cell 7 (`E2B_ADAPTER_*` set) | the full cross-model line; optional |
| P6 | live GPU re-verify of the serve fixes (move coercion, `<` whiff, result-echo) | any chat cell | confirms the answer-quality fixes; CPU-tested, not GPU-verified |

**If you only have time for one thing: run P1 + P2.** They convert the two weakest/most-confusing
artifacts (13.5% completion, buggy chats) into the real story.

---

## 7. One-line honest status

The defensible, presentation-ready results are **routing 88.7% vs 49.6% base** and **OOD completion
91.7%/95%**. The chess-completion number and the showcase chats shown earlier are **not yet valid** (a
harness artifact and a config bug, both fixed) and must be re-run (P1, P2). The cross-model line chart
and GGUF-quant comparison are **incomplete** (P3-P5).
