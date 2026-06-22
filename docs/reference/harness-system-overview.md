Parent: none

# Harness system overview — peer-review packet (as of 2026-06-22)

**Purpose.** A single self-contained description of the whole system — product thesis, the
trained model, the harness contract, the serving loop, the training corpus, the robustness
layers, and the latest evaluation evidence — written so an **external reviewer** (human or AI)
can judge the work and surface gaps without reading the repo. Each section links to the deeper
reference doc for those who do have the repo.

**How to review this.** The claims that matter are in §3 (contract), §6 (eval evidence), and §7
(known gaps + open questions). §7 is written adversarially against our own work *on purpose* —
please pressure-test it and add what we missed. Where a number is stated, its source artifact is
named so you can check it.

**One-line status.** A frozen Gemma 4 E4B QLoRA adapter that operates a general skill+tool harness;
on held-out validation it routes skill/tool/none at **96.4% verb accuracy vs 82.9% for the
untrained base on the identical harness** (exact-name 73.9% vs 17.6%). The model is frozen; only the
serving harness is still being changed.

---

## 1. What the product is

The product is **not** a chess engine and not a chatbot. It is a **general agentic harness
operator**: an LLM that, given a per-request list of skills and tools in its system prompt, **chooses
among them and reasons to complete a goal in any domain**, narrating tool results without computing
them. Chess-coach is the flagship demo domain (~25% of the corpus); the other ~75% is general
harness operation across arbitrary skills/tools. The thesis under test: *a small (4B) model can be
trained to operate an open-ended, in-context tool/skill surface reliably* — i.e. routing is a
learnable, transferable skill, not per-domain memorization.

Design rationale: `docs/reference/2026-05-23-chess-coach-sft-design.md`. Terminology:
`docs/reference/glossary.md`.

## 2. The trained model

| property | value | source |
|---|---|---|
| base | Gemma 4 **E4B** (≈8B, instruction-tuned), 4-bit (QLoRA) | `CLAUDE.md` repo map |
| adapter | LoRA **r=16, α=32, dropout 0**, **all-linear** (q/k/v/o + gate/up/down + per-layer gates), language/text modules only (vision frozen) | `results/gemma4_chess_e4b_kaggle/best/adapter_config.json` |
| trainer | **Unsloth** Gemma4-native QLoRA, `max_seq=1664` | `src/llm/llm_training/run_train.py` |
| train HW | Kaggle **T4** (16 GB), single-GPU | memory `e4b-needs-2xt4-balanced` (RESOLVED: fits 1 T4) |
| loss | up-weighted ("FORMAT_WEIGHT") on the control tags **and** skill/tool **names** | memory `e4b-attn-only-freegen-collapse` |
| serve | HF nf4 (full fidelity) or **q4_0/Q5_K_M GGUF** locally on an RTX 4060 (8 GB); vision retained via mmproj | `CLAUDE.md`, memory `gguf-keeps-image-reading` |
| decode | greedy, temperature 0 (penalties OFF — they corrupt name copying) | memory, commit `09d209eb` |

**Version history (the "why we retrained" arc — measured trend is built but not yet re-run):**
- **v2** (2026-06-17): attn-only LoRA → could not emit the harness format at all. Fix → all-linear.
- **v3** (2026-06-18): format correct but skill/tool **names** corrupted on copy. Fix → up-weight
  control tags in the loss + decode penalties OFF.
- **v4** (2026-06-19, current): extend the loss weight to **names** + longer training + hardened base
  harness. This is the benchmarked adapter (`RyanDev1st/gemma4-chesscoach-ckpt-v4`).
- A prior **E2B** production adapter (attn-only r=8) exists as the generation-before baseline
  (`RyanDev1st/gemma4-chesscoach-e2b`, private). Caveat: older training contract → its score on the
  current contract is "old weights on today's task," a valid but specific comparison.

## 3. The harness contract (the heart of the system)

Rendered **identically at train and serve** by `build_system(...)` in
`src/llm/llm_training/system_prompt.py` (single source of truth; ~1062 tokens). The model conditions
on the exact surface it is allowed to use, listed per request — **the list is the authority, not the
model's memory.** Two verbs, **exactly one action per generation step**:

- `<skill>NAME</skill>` — load a listed skill's *guidance* (instructions, not a function). Its body
  is delivered as a tool result (progressive disclosure — only names+descriptions are in the prompt;
  bodies load on demand). There is **no** `load_skill` tool — the verb is the mechanism.
- `<tool>NAME arg=value</tool>` — call a listed tool to get data or change state; returns a result.

These two tags are the **only** action formats (no JSON, no other function syntax). Reasoning runs in
four modes selected by a prompt signal (verbatim text in `system_prompt.py::_REASONING_LINE`):
- **fast** — no `<goal>`, no `<think>`; act and answer directly.
- **think** — `<goal>` once, then a brief `<think>` before every step.
- **auto** — `<goal>` once, then `<think>` only before a *hard* choice (interleaved).
- **plan** — `<goal>` per objective, a `<plan>` checklist of steps, then do every box in order before
  the final synthesis.

`<goal>`/`<plan>` render to a UI panel; `<think>` is hidden from the user. A customization overlay
(`agent_overlay`) shapes tone/persona only and never overrides the harness (empty by default → no
train/serve drift). Full account: `docs/reference/harness-architecture.md`.

## 4. The serving loop

`CoachLoop.respond` in `src/llm/backend/inference.py`. One turn:

1. Build the system prompt (contract + memory block) + windowed history + the user turn.
2. **Generate** until an action close tag — `ACTION_STOP = ["</tool>", "</tool_code>", "</skill>"]`
   (the missing `</skill>` stop token was the cause of an auto-mode double-emit bug; now fixed).
3. **Extract the first action** and execute it (skill load or tool call) via `ToolExecutor`
   (`backend/tools.py`). Append the result to the convo as DATA.
4. Repeat until the model emits a no-action final reply.
5. **Finalize**: `_split_reasoning` lifts `<think>`/`<goal>` out of the visible bubble; `is_plan_panel`
   decides whether a `<plan>` renders as a panel vs a normal reply.

**Deterministic robustness layers** (serve-side, kept minimal — they should *route/repair*, not do the
model's job; this is a key review point, see §7):
- corrective tool errors with the right verb named (e.g. a tool called as a skill → "is a tool, not a
  skill — call it with `<tool>…`"; **symmetric in both directions** as of the latest fix) so the loop
  self-corrects instead of dead-ending.
- `_is_deflection` + `_force_answer`: if a no-tool turn deflects with a generic capability blurb, force
  a real grounded answer.
- `_verify_fulfilled`: a gated self-check that the reply's claims trace to a tool result.
- `_correct_eval_number` / `_correct_move_names`: number/name consistency guards (chess-specific
  grounding backstops).
- KV-cache prefix reuse across loop steps (pure prefix-extension only — Gemma's sliding-window cache
  cannot be cropped; A/B self-check + no-op fallback → worst case no speedup, never wrong).

Robustness design history: `docs/reference/2026-06-12-coverage-reliability-design.md`,
`docs/reference/harness-strengthening.md`. Latest serve fixes are tracked in memory
`harness-serve-hardening`.

## 5. The training corpus (v1_2)

Built by `src/llm/llm_dataset/v1/` (spec = `contracts.py`, writer = `profiles.py`). Measured
composition (from `data/sft/v1_2_{train,val}.jsonl.gz`):

- **72,329 train / 2,731 val rows**, 31 slices (11 chess "letter" slices A–K + 20 general "V1_"
  slices). Held-out val is a true split.
- Reasoning-mode mix (train): fast 25,048 / auto 27,609 / think 17,471 / plan 2,201.
- Design mix ~**75% general / 25% chess** (chess also appears inside several V1_ slices, so we do not
  assert a measured domain split).
- Rows are assembled combinatorially from cards × slices, grounded against real backends (Stockfish,
  the `python` verification tool, real plugin executors), and passed through a quality gate
  (`scripts/final_corpus_audit.py`) enforcing a token-length ceiling and final-answer diversity.

The generalization claim (route by reading the in-context description, not by memorizing phrasings) is
the corpus's central design goal — see `docs/reference/sft-corpus-generation.md` and memory
`chess-sft-generalization-not-phrase-memorization`. **This is a primary thing to peer-review** (§7).

## 6. Evaluation — methodology and latest evidence

**Harness:** `src/llm/llm_training/eval_benchmark.py` (+ `bench_report.py`, `bench_misses.py`,
`bench_suites.py`, `bench_transcript.py`). Routing = a 3-class decision on the **first action** (skill
/ tool / none); we report a confusion matrix, per-class precision/recall/F1, exact-name accuracy,
format validity, throughput, per-slice accuracy, and a per-row **miss analysis** (what the model
actually emitted on each miss — wrong-name vs wrong-verb). Two conditions share the identical harness:
**adapter** (the product) vs **base** (same E4B, LoRA disabled via `AdapterView` — isolates what the
SFT weights bought). Fast mode is forced for throughput (≈13.7 s/row on T4, HF nf4, unmerged LoRA).

**Latest measured results — validation, n=692 held-out (2026-06-21 Kaggle run):**

| metric | adapter+harness | base+harness | (base, no harness) |
|---|---|---|---|
| verb accuracy | **96.4%** | 82.9% | 3.0% |
| macro precision | **78.3%** | 46.2% | 1.0% |
| weighted precision | 97.7% | 92.0% | — |
| exact-name | **73.9%** | 17.6% | 0.0% |
| format validity | 99.7% | 93.5% | — |

Per-class (adapter): skill recall 0.97 / prec 1.00 (629/646); tool recall 0.76 / prec 0.86 (19/25);
none recall 0.90 / prec 0.49 (19/21). Base `none` recall is only 0.33 — the base barely learned *when
not to act*. The no-harness base collapses to "none" everywhere (the protocol is taught, not latent).

**Held-out STRESS, n=20** (messy/slang phrasing + unseen out-of-domain catalogs + decline cases):
adapter 90% verb, base 95% (a 1-row swing — **not significant at n=20**), no-harness base 25%.

**Two methodological points that make the result a lower bound (not cherry-picking):**
1. **First-action scoring undercounts production accuracy.** Many "misses" are tools emitted as
   `<skill>` (E→`best_move`, A→`move_san`, H→`list_pieces` — all tools). At serve the harness's
   symmetric corrective error redirects these to `<tool>` and the loop recovers — **proven end-to-end
   in the captured transcript** (a metronome and a breathing request both self-correct and answer).
2. **Forcing fast mode handicaps the adapter on its own trained slices.** Slice G read 0% only because
   forcing fast collapsed the trained `goal→skill` sequence (the miss log shows it emitted the goal
   keyword `<skill>threats</skill>` instead of reaching `<skill>chess-coach</skill>`). A native-mode
   probe (score each row in its trained mode) is built (`--native-mode`, notebook Cell 6.5);
   mechanism proven offline, **magnitude pending the rerun.**

Full read + the miss tables: `docs/findings/2026-06-21-routing-benchmark-interpretation.md`.

## 7. Known gaps & open questions (the review target)

**Gaps we already see (please confirm / extend / refute):**
1. **Eval is single-turn, first-action only.** The corpus has multi-turn and compound-plan slices, but
   the headline benchmark scores only the first routing decision. We have **no measured multi-turn task-
   completion metric.** Is first-action routing a sufficient proxy for agentic quality?
2. **The deterministic backstops may flatter the model.** `_force_answer`, `_verify_fulfilled`,
   `_correct_eval_number/_correct_move_names`, and the corrective verb errors all repair model output at
   serve. We believe they *route/repair* rather than *substitute*, but an external reviewer should check
   they aren't doing the model's job and inflating apparent quality. Where is the line?
3. **Small support on the non-skill classes.** tool n=25, none n=21 in val → wide confidence intervals;
   tool recall 0.76 is the weakest real spot. Macro-precision is the conservative headline; raw accuracy
   over-weights the dominant skill class (646/692).
4. **Stress n=20 is too small** for strong robustness claims.
5. **Pending comparisons:** native-mode numbers and the E2B-vs-E4B generation comparison are built but
   not yet measured; the v2→v3→v4 trend is built but not re-run (the prior run OOM'd loading v3 — fixed).
6. **Greedy-only decoding.** All evals are temperature 0. We have no measurement of robustness under
   sampling, so brittleness there is unknown.
7. **Frozen-weights failure modes (un-fixable without retraining):** arg-extraction deflection (asks back
   instead of filling args from the schema) and training-domain gravity (on an out-of-domain *failure* it
   tends to fall back to chess). Both are documented, not yet quantified.
8. **Memory system** is single-user, auto-capture behind a heuristic write-discipline gate; the gate's
   precision/recall is not formally evaluated.
9. **Generalization vs memorization.** Slices are template-assembled; the corpus has a diversity pass to
   avoid phrase memorization, but a reviewer should judge whether held-out val truly tests generalization
   or shares too much structure with train.

**Open questions we'd most like an answer to:**
- Is the **96.4% vs 82.9%** the right framing, or does the dominant skill class make verb-accuracy a
  weak headline — should we lead with macro-precision (78.3% vs 46.2%) or exact-name (73.9% vs 17.6%)?
- What's the **minimum additional eval** that would make the "the agent works" claim defensible — a
  multi-turn completion suite? a held-out *unseen-domain* routing set larger than n=20?
- Are the deterministic serve layers a legitimate part of "the product," or should the model be evaluated
  **without** them to report its intrinsic capability separately?
- Is training **language-modules-only** (vision frozen) leaving measurable capability on the table for a
  text-routing task, or is it correctly scoped?

## 8. Where to look (repro map)

- Contract: `src/llm/llm_training/system_prompt.py` · Loop: `src/llm/backend/inference.py` · Executor:
  `src/llm/backend/tools.py` · Skills/plugins: `src/llm/skills/`, `src/llm/backend/plugins/`.
- Corpus generator: `src/llm/llm_dataset/v1/` · Corpus: `data/sft/v1_2_{train,val}.jsonl.gz`.
- Trainer: `src/llm/llm_training/run_train.py`, `train_cuda.py` · Eval: `eval_benchmark.py`,
  `eval_confusion.py`, `bench_*.py`, `report/` · Kaggle: `kaggle_benchmark.ipynb`.
- Deeper references: `harness-architecture.md`, `sft-corpus-generation.md`, `harness-strengthening.md`,
  `2026-06-12-coverage-reliability-design.md`, `2026-06-09-context-window-system.md`.
- Latest eval read: `docs/findings/2026-06-21-routing-benchmark-interpretation.md`.

Run the benchmark: `python -m llm_training.eval_benchmark --adapter <best> --per-slice 25` (val) /
`--suite stress` / `--native-mode --slices G,H,…` (fair mode-dependent probe). Tests:
`python -m pytest src/llm/backend src/llm/llm_training -q`.
