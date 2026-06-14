# Handoff — general agentic harness (Gemma E4B), corpus + next architecture

Plan of record: `implementation.md`. This file = current state, what was just done, the
architecture we're considering next, and candid notes. Written 2026-06-14, branch
`feat/chess-coach-sft` (not pushed). A fresh agent should be able to act from this alone.

---

## ⚡ SESSION UPDATE 2026-06-15 (supersedes everything below — read FIRST)

**ALL THREE STAGES BUILT + REGEN DONE + GATE PASS + CORPUS COMMITTED. TRAINING-READY.**

- **Stage 1 (plan mode):** thinking-mode `<goal>` anchor — committed ONCE per turn (think/auto/plan), promoted OUT of per-step `<think>` so it's seq-neutral; fast stays bare. Plan mode commits EVERY goal of a compound request. Serve de-risk shipped: the loop surfaces a `<goal>`/`<plan>` panel turn and KEEPS working the boxes (was a CRITICAL trap that made plan mode worse-than-fast), + a plan-box backstop that steers the model to fill an unfilled box before finalizing (generalizes to any domain). Clever context compaction pins the goal/plan on eviction. Early-stop-reduction eval built (`llm_training/eval_early_stop.py`) to measure at E4B whether the anchor cuts half-answers.
- **Stage 2 (audit):** `V1_T_audited_plan` — a self-authored checklist whose tool-checkable boxes are CLOSED by running the `python` tool + reading stdout, NEVER asserted. Loads a `plan-audit` skill (the procedure). Split-determinism: semantic boxes stay SOFT (not tool-audited — no audit theater). Honest-partial loop-cap. New gate `audit_boxes_grounded` rejects assert-without-audit. SEQ go/no-go PASS (max 1320<1664 in samples).
- **REGEN (commit 008a01df):** full 75,060-row v1_2 with Stages 0/1/2 + goal anchor. Split train=72,942 / val=2,118 (final-text de-leaked). `scripts/final_corpus_audit.py` GATE = **PASS**: over_seq=0 (max **1642**<1664 — the tight V1_P/F tail held at full scale), template_fallback=0, tool_result_missing=0, leakage=0, reasoning_mode 0/0, validate_failures=0. Mode dist fast 25.5k/think 18.3k/auto 28.9k/plan 2.4k. Trainer loads the .gz via the .jsonl path; all 72,942 train rows load through `load_jsonl_chat`.
- **QUALITY FIX (2026-06-15):** dropped goal substitution from `think_answer` — it inlined verb-phrase goals as the object of "answer" → "time to answer decide between the options directly" (broken grammar) in ~10,064 trained rows (hidden `<think>`, serve-stripped but trained-on). Now goal-free (objective already in the leading `<goal>`); broken-grammar think now **0 of 108,422** think turns.
- **VAL-COVERAGE FIX (2026-06-15):** the think fix made answer-think uniform → more final-text collisions → blanket de-leak emptied val for low-diversity slices (val coverage dropped to 11/31 — chess + V1_Q etc. had 0 val, blinding post-train per-slice `eval_routing`). Fixed `build.split_train_val` with a per-slice VAL_FLOOR=12: keep all clean val rows, then top each slice up to the floor from NON-exact-dup overlapping rows, then drop any train row that's an exact twin of a kept val row. Relaxed the gate's `val_final_text_in_train` from hard-fail to a tracked metric (the real leak guard `val_exact_row_in_train` stays hard-gated at 0; routing eval cares about the first-turn tool, not final text). Result: **30/31 slices in val** (V1_Q correctly excluded — ~10 unique full rows, un-holdout-able), chess all covered (12+ each).
- **FINAL CORPUS = latest commit. train=73,396 val=1,661.** GATE PASS: over_seq=0 (max **1634**<1664), val_exact_row_in_train=0, template_fallback=0, tool_result_missing=0, reasoning_mode 0/0, validate_failures=0; tracked final-overlap=234 (the floored rows). THIS is the corpus to train on.
- **FULL EPOCH = ~30h / ~3 SESSIONS (DECIDED 2026-06-15).** Real wall-clock: 1 epoch = 2283 updates (DDP-2) × grad_accum 16 × ~2.9s/micro ≈ **~30h** > Kaggle's 12h session ceiling, so 1 epoch spans ~3 sessions (~640 updates each). (Earlier "finishes in ~3-5h" was WRONG — it used the grad_accum=1 fit-test s/step.) User chose a FULL epoch over a one-session partial (~800 steps), so multi-session resume is mandatory.
- **OFF-KERNEL HUB CHECKPOINT (commit 06a56991) — the failsafe that makes the 3-session epoch safe.** A 12h timeout is a **SIGKILL**, not a Python exception → the in-process crash-save can't catch it, and a timed-out *committed* run's local `/kaggle/working` Output is NOT guaranteed to persist. So `train_unsloth._hub_push` mirrors `checkpoint/` (every SAVE_EVERY=50, ~40 min) + `best/` (on val-improve) to a **private HF Hub repo**, OFF the kernel — bulletproof across sessions AND accounts. Rank-0 only (under existing `if main`). Opt-in via `CHESS_CKPT_REPO`; best-effort (upload hiccup never crashes training). Worst case lost = ≤50 steps between pushes, never a whole session.
- **RUN IT (`kaggle_e4b_qlora.ipynb`, MODEL=gemma4_e4b, DATA_STEM=v1_2, MAX_SEQ=1664; the user uses a Save&Run-All COMMIT so it survives tab-close):**
  - **Session 1 (fresh):** Cell 1 → set `CKPT_REPO="<hf-user>/gemma4-chesscoach-ckpt"` (auto-created private), `RESUME=False`. Run All → trains ~12h to ~step 640 → SIGKILL. Checkpoints already on the Hub. Confirm in logs: `[hub] pushed checkpoint -> <repo>` (if absent, CKPT_REPO wasn't set — fix before trusting it).
  - **Sessions 2 & 3 (resume, any account):** Cell 1 → same `CKPT_REPO`, `RESUME=True`. Run All → Cell 6.6 pulls latest `checkpoint/`+`best/` from the Hub (single in-kernel download, before `accelerate launch` → no DDP race) → Cell 7 continues from saved step. S2 ~640→~1280, S3 ~1280→2283 done → final save + Cell-8 zip.
  - **Retrieval (committed run):** Output commits only when the run ENDS (success or kill); `best/` is on the Hub from S1 on, so you can stop early and serve a partial-epoch adapter (scout: behavior at ~250 steps). On a clean finish Cell 8 zips runs/OUTPUT.
  - **DDP go/no-go = Cell 6.55 A/B** (real 2-proc trainer, 8 steps). Unverified point: `acc.prepare()` DDP-wrapping the Unsloth 4-bit + gradient-checkpointed model (can need find_unused_parameters / static_graph). A/B OK → **DDP=True, MAX_STEPS=2283**. A/B errors or 1 GPU only → **DDP=False, MAX_STEPS=4566** (and ~60h/~5 sessions — Hub resume still carries it).
  - **Token:** Cell 5 `login(HF_TOKEN)` caches creds for the push/pull; HF_TOKEN needs **write** scope or pushes log `[hub] push skipped (auth…)`.
  - Legacy Dataset-zip resume still works (Cell 6.6 falls back when `CKPT_REPO=""`).
- **IMAGE READING RETAINED (verified 2026-06-15, commit 7f08976f):** text SFT doesn't cost vision — base frozen, vision tower untouched. Bundled llama.cpp (build 9295, `mtmd.dll`) supports Gemma-4 vision via mmproj (`Gemma4VisionAudioModel` → GEMMA4V/GEMMA4A; runtime decodes gemma4v). `export_gguf.py` now emits the mmproj off the BASE. Serve+image: `llama-mtmd-cli -m <text.gguf> --mmproj mmproj-gemma4-vision-f16.gguf --image pic.jpg -p '…'`. UNTESTED: actual image-in→out quality post-SFT (no image rows/eval) — run one smoke image to prove it. See memory [[gguf-keeps-image-reading]].
- **POST-TRAIN (no retrain):** serve mode-trigger (a complexity router to set `reasoning_mode="plan"` on multi-step asks — currently `respond()` passes ""), goal/plan panel UI payload, and run the early-stop eval. KNOWN corpus limitation (acceptable for v1, not gated): fixed-lesson slices V1_Q/I/J/C/G have low final-diversity (V1_Q 10 distinct/2385, "Paris…" 266×) — deliberate one-lesson slices, 3% of corpus, lesson generalizes; add paraphrase pools if a later eval shows phrase-memorization. Code-debt: `audit.py BASE_UNIVERSAL_TARGET=257` stale (20 V1_ slices) — reconcile at next regen; `final_corpus_audit.py` is the real gate, unaffected.

---

## ⚡ SESSION UPDATE 2026-06-14b (older — superseded where it conflicts with the block above)

Big deltas since the body of this file was written (the older sections below are partly
out of date where they conflict with this block):

- **TRAINING WORKS ON FREE KAGGLE.** E4B 4-bit QLoRA fits **ONE** T4 at seq 1664 — the old
  "DDP OOMs → single-GPU" / "needs Colab Pro" claims are WRONG. The whole OOM saga was a
  self-inflicted broken-gradient-checkpointing bug (device_map=balanced hooks disabled
  checkpointing; fix = `enable_input_require_grads()` + no device_map). See memory
  [[e4b-needs-2xt4-balanced]] (RESOLVED) + [[qlora-single-gpu-oom-playbook]].
- **DDP NOW WORKS** for ~1.82× on 2×T4 (`accelerate launch --num_processes 2 --multi_gpu`),
  gated add-on; single-GPU byte-identical. Memory [[e4b-2xt4-three-miracles]] + the playbook.
- **SCOUT PASSED (Stage 0 validated):** 250-step E4B scout served on T4 → verification-as-
  tool-use, fast/think/auto gating, skill routing all EMERGED. [[scout-e4b-behaviors-emerged]].
- **Multi-session resume + lean subset + 250-step de-risk** all built (the "3 miracles").
- **Stage 0 SHIPPED** (python verify tool + V1_R). **Stage 1 BUILT** (this session):
  `plan` reasoning mode + `V1_S_compound_plan` (goal-driven anti-early-stop loop), seq
  max 1243, validates clean, NOT yet regen/trained. See [[stage1-plan-mode-built]].

**Plan of record now (deadline ~1 week): build all stages into the DATA, then ONE train.**
1. Stage 2 (audit skill verifies boxes via the python tool) — IN PROGRESS / next.
2. Then ONE regen + `final_corpus_audit.py` gate + the full E4B train (Kaggle, multi-session).
3. Serve-side (post-train, no retrain): plan box-tracking gate + the plan PANEL UI.

**Two standing user requirements (2026-06-14b):**
- **`<goal>` ALWAYS-ON** — every request starts with `<goal>` to force intent-understanding,
  not only in plan mode. OPEN: every-row (big regen + seq cost) vs multi-step-only — DECIDE
  before regen.
- **Plan mode in the ACTUAL serve harness** — the runtime thinks to build the checklist +
  the deterministic gate blocks the final until boxes clear. (Training data exists; serve
  gate is the post-train task.)

---

---

## 0. What the product is (don't lose this framing)

A **general agentic HARNESS operator**: an LLM that **chooses among the skills+tools listed
in its prompt and thinks to complete a goal in ANY domain**, narrating tool results without
computing them. Chess-coach is the **flagship demo domain, one of many** — NOT the whole
product. Corpus mix is ~75% general / ~25% chess.

**Contract** (`src/llm/llm_training/system_prompt.py`, `BASE_HARNESS`): TWO verbs, one action
per step —
- `<skill>NAME</skill>` loads a listed skill's body into context (progressive disclosure;
  catalog of name+description always present, body pulled on demand, persists).
- `<tool>NAME args</tool>` calls a function. **There is NO `load_skill` tool anymore** — we
  dropped it for the `<skill>` verb (clearer separation: skill = guidance you read, tool =
  function that runs).

**Reasoning modes** (prompt-signal gated, so fast-vs-think is a real toggle, not always-on):
`fast` (no `<think>`), `think` (`<think>` every step), `auto` (`<think>` only on hard
decisions — interleaved). Distribution in corpus: fast 26.3k / think 18.9k / auto 29.8k.

**Train==serve:** one `build_system(skills_index, tool_manifest, plugin_context,
reasoning_mode)` renderer produces the system prompt at both train and serve. Gemma's chat
template silently drops `role="tool"`, so `chat_format.remap_tool_messages` rewrites tool
turns to `<tool_result>`-wrapped user turns at BOTH train and every serve path — without it
the model is blind to tool results and fabricates. This is load-bearing; see memory
`gemma-template-drops-tool-role`.

**Infra:** train E4B QLoRA via **Unsloth** (anomaly-guarded; load `unsloth/gemma-3n-E4B-it-
unsloth-bnb-4bit`, NOT a raw QAT checkpoint — see memory `unsloth-load-anomaly-e2b`) on
**Kaggle/Colab T4**, seq **1664** → LoRA adapter → serve **q4_0/Q5_K_M GGUF locally on the
RTX 4060 (8 GB)**. E2B is the currently-shipping fallback; E4B is the target and is wired
(`src/llm/llm_training/kaggle_e4b_qlora.ipynb`, `colab_e4b_qlora.ipynb`). DDP on 2×T4 OOMs →
ship single-GPU (memory `ddp-not-viable-e2b-t4`).

---

## 1. What this session did (all committed, branch not pushed)

Goal of the session: a "cynical" final pass on the v1.2 corpus before training — catch
anything that would force a retrain/repatch/regen later. It found and fixed real defects.

### 1a. Built a permanent full-corpus gate — `scripts/final_corpus_audit.py`
Measures the things the per-row validator and the sampled audit never checked, over **all
75,060 rows through the EXACT training path** (`build_system` + `remap_tool_messages` + the
real `gemma4_e2b` tokenizer at `src/llm/models/gemma4_e2b`). Exits non-zero on any
**retrain-forcer**:
- real token length > MAX_SEQ (1664) → would silently truncate finals
- chat-template fallback (a row that breaks Gemma's template → format drift)
- tool-result not surviving the remap (model trained blind)
- train↔val leakage (exact row, exact final text)
- reasoning-mode integrity (fast-with-`<think>`, think-without-`<think>`)
- per-row validation failures
It ALSO reports (not gated) **final-answer diversity** — overall distinct ratio + worst-8
slices by distinct/rows — so canned-final regressions are visible. Run it before trusting
any regenerated corpus. **Run from repo root** (`python scripts/final_corpus_audit.py`).

### 1b. Fixed the seq-truncation surprise (the retrain-forcer the sample audit missed)
234 rows tokenized past 1664 (max 1715), ALL `V1_G_multi_tool_budget`: in `think` mode every
rote step of its engine chain carried its own `<think>`, stacking ~120 tok over the ceiling.
The trainer hard-truncates at `max_len` → those rows trained on a CHOPPED final. Fix: new
`execute` step-kind in `renderer/thinking.py` (`_NEVER_THINK_KINDS={"execute"}`) that never
emits `<think>` in any mode — a budget task reasons ONCE up front, then executes silently.
Trimmed V1_G chain 4→3 calls. Max now 1653, 0 over. Memory `seq-ceiling-v1g-truncation`.

### 1c. Grounding: derived finals from per-row engine scenes (`renderer/synth_engine.py`, NEW)
The universality chess-tool slices (V1_G/I/E/H) used canned constants — every V1_G said
"+0.15 / Nf3 tops" regardless of the (also constant) tool result. That trains narration to be
INDEPENDENT of tool output — the opposite of grounding. Now a seeded per-row scene produces
varied eval/best-move/board_state in the live-backend format, and the final is DERIVED from
that row's actual numbers. Added `narration_grounded` to V1_G's rules to enforce the copy.

### 1d. Killed catastrophic final-answer repetition (the real ship-blocker)
A diversity audit found: **V1_O (25% of corpus, 18.8k rows) had only 40 distinct finals**
(one repeated ~470×), and seven slices (V1_N/M/D/A/B/K/J, chess C/J) had **exactly ONE**
constant final repeated thousands of times → memorization risk. Fixes (varied the GROUNDED
outputs, NOT the prompts — fake user phrasings risk memorization with no generalization gain):
- `domains.py`: `Domain.scenes` = tuple of `(call, tool_result, finding)`; `skill_routing`
  picks a scene + a seeded guiding closer (`CLOSERS`). Synthetic topic pool 20→40.
  Back-compat `.call/.tool_result/.answer` properties kept.
- `universality_prompts.FINAL_POOLS`: 7-paraphrase pools for the 1-final V1_* lesson slices;
  `_final` appends a seeded closer where natural (NOT the V1_J greeting) → ~70 distinct/slice.
- `finals.py _LESSON_FINALS`: pools for chess B/C/H/J (C/J stay statements per the leadin
  contract test); B/H get closers.
- `chess_kb.py`: `KBItem.answer` (str) → `answers` (3 paraphrases) + `pick_answer`; topic
  stays correct, distinct 4→12.

**Result:** overall distinct finals 13.3% → 15.5%; worst single repeated final **2778 → 273**;
every slice now ≥5 distinct (V1_O 40→~990). Memory `sft-final-diversity`.

### 1e. Earlier in the session (also committed)
- QC round: topic-keyed chess I/K (was confidently wrong answers), grounded V1_I eval,
  real V1_F board_state from the FEN, real V1_L coaching final, retrieval-shaped synthetic
  tools, new `V1_Q_no_skill_direct` slice (teaches plain speech / decline when no skill fits).
- Fixed V1_Q `<think>` grammar (`think_direct`/`gated_direct`, replacing a goal-substituted
  template that read "answer reply directly … directly").

### Current corpus state — TRAINING-READY
- `data/sft/v1_2/{accepted,rejected}.jsonl.gz` (75,060 / 7,500) + `v1_2_{train,val}.jsonl.gz`
  (train **73,175** / val **1,885**).
- **GATE: PASS** — over_seq 0, template_fallback 0, tool_result_missing 0, val exact/final
  leak 0, reasoning integrity 0, **validate_failures 0/75,060**. token max **1653**.
- 28 slices. Last commit `72f8055e`.
- Human-inspectable sample: `docs/2026-06-13-v1.2-random-sample-inspection.md` (10 truly-random
  rows/slice, real token lengths; regenerate with `python scripts/make_sample_doc.py`).

### Honest quality rating (mine, end of session)
Contract integrity 9/10 · grounding 9/10 · behavioral coverage 8/10 · answer diversity 3→**8**
(was the real hole) · naturalness 6→**8**. Overall **~8.7/10**, training-ready, no memorization
worry. Remaining low-diversity slices (V1_Q's 10 fixed greeting/decline pairs, chess C/J
statement-contract, low-row G) are **bounded by design, not defects** — pushing them further
is the "do things that don't help" trap the user warned against.

---

## 1.5 Stage 0 — BUILT + GATED (this session, committed; NOT yet trained)

Stage 0 from §2/§8 is now in the corpus. The keystone is a real **`python` tool** the
trained model calls to **verify a claim by running a script and reading stdout** — the
way Claude verifies instead of fabricating (NOT a calculator front-end; that was the
first wrong cut, corrected on user feedback).

- **`src/llm/backend/sandbox.py`** — `run_python(code)`: isolated `python -I` subprocess,
  3s timeout, code/output caps, temp cwd. Returns `output: …` / clean `error: …`. Security
  posture documented in-file: contains hangs/crashes/floods; NOT a boundary vs hostile code
  (`-I` still exposes stdlib) → fine for the LOCAL single-user 4060 demo; OS-sandbox before
  any untrusted/multi-user exposure. Executor wired in `backend/tools.py` (`python` →
  `run_python`); served via `serving_tool_manifest` + `_TOOL_NAMES`.
- **Free-text `code=` arg** — scripts have spaces, so `code=` captures the rest of the call,
  mirroring `query=`/`fen=`. Wired in BOTH `backend/toolfmt.parse_call` AND
  `llm_dataset/v1/validate._parse_args` (train==serve parse agreement; the gate's per-row
  validation depends on it).
- **`V1_R_compute_grounding` slice** (`renderer/compute.py`, **990 train / 105 val**, plan=80
  base — bumped from 30 after the 2026-06-14 audit: ~28 examples per (3 modes × 12 families)
  cell, the robust-learning threshold for a 4B; held-out val 94/105 = 89.5% novel prompts):
  **~70% verify-then-claim** (user asks a judgment — "am I averaging above 85?" — model runs
  the script, reads the value, asserts the GROUNDED verdict) + **~30% compute-on-request**
  (raw number). The verify-then-claim shape is box-auditing in miniature → seeds Stage 1/2.
  No domain skill fits → tool-direct. `<think>` + tool description are **verification-forward**
  ("verify before I claim it, not guess"). Grounding enforced by the existing
  `narration_grounded` gate (every two-decimal number in the final ⊆ the tool's stdout).
- **Plug-and-play calculator template** (user ask): `catalog.CALC_TEMPLATE = print(f"{EXPR:.2f}")`
  — ONE known-good snippet, single-sourced, surfaced in the tool description AND used verbatim
  by the renderer, so a weak coder model substitutes the expression instead of composing code.

**Verified:** sandbox 5/5, compute 9/9 (incl. real-subprocess exec match), my-change dataset
tests 38/38, full dataset suite 111. **GATE: PASS** (validate_failures 0/75,064, over_seq 0,
all 8 gate fields 0). Serve path executes end-to-end (`output: 12.96`, matches train render).
Full corpus audit `docs/2026-06-14-v1.2-corpus-audit.md`: 0 TRUE full-example dups, mix
75/25, fast/think/auto think-ratio 0.000/0.575/0.969 (three distinct behaviors, all modes in
every slice). Caveat: val is a same-distribution loss probe (~92% prompt overlap, structural) —
gauge generalization via `eval_routing.py`+serve, NOT val loss (V1_R is the held-out exception).

**The seq number (handoff §8 go/no-go, MEASURED):** a single python-verify chain tokenizes to
**max 1469** (p99 1450; fast 1357 / think 1444 / auto 1469) vs the 1664 ceiling — Stage 0 fits
with ~195 headroom. Corpus max unchanged at **1653**. **Implication for Stage 1:** the
contract+manifest floor is ~1255–1290 tok; ONE audited step (think + tool + result) adds
~130–180. So only **~2–3 audited boxes fit 1664** before it blows — Stage 1's multi-box chains
will be seq-tight; measure each realistic chain before training (this is the real constraint the
§3 Colab table keys off).

**What Stage 0 still needs (NOT done): TRAIN it.** Run the E4B QLoRA notebook on Kaggle, serve,
and measure the empirical unknown that gates everything in §2: **does E4B reliably
call→read→narrate the computed value (not fabricate)?** If no → the agentic architecture can't
stand. If yes → we've shipped a fabrication fix worth having alone, and Stage 1 is unlocked.

---

## 2. The architecture we brainstormed next — "truly agentic E4B" (NOT yet built)

User's question that opened it: *does the model know to call MULTIPLE skills and tools in one
run?* Measured answer from the corpus:
- ≥2 tool calls in one run: **30%** of rows (good; up to 3 tools).
- ≥2 DISTINCT skills in one run: 10% — **but every one is the same pattern** (`hood-human-chat`
  normalize → one domain skill). **3+ distinct skills: 0%.**
- So: good multi-TOOL depth + a normalize→route bridge, but **no composition of multiple
  DIFFERENT domain skills for a compound goal** ("review this diff AND explain why its query
  is slow"). That's the gap between "good tool-caller" and "truly agentic."

### The idea (user, paraphrased + sharpened)
Lay out the goal, let the agent **loop until it clears a checklist that accomplishes the goal**:
1. Model receives prompt → **defines the goal itself**, sets it as priority.
2. Lays out **checkboxes** (sub-goals / acceptance criteria) it authors.
3. **Audits the checkboxes** — but NOT by free-reasoning. The key clarification: the audit is
   driven by a **skill** (a procedure, like the superpowers SKILL.md files Claude reads), and
   each box is verified by **running a real tool — calculator / python — and reading the
   output**, not by asserting. "The model doesn't just reason; it runs the script, inputs the
   values, checks the outputs." Compute/verification is **offloaded to executors** so the model
   can't fabricate the audit values.
4. Loop until each box is addressed, **narrating naturally** to the user throughout.

### My (scientist) analysis and the calls I made
**Name:** self-authored checklist + verification-as-tool-use + bounded loop. Sits at
Plan-and-Solve ∩ ReAct ∩ Reflexion, but with the good twist that **the model proposes progress
and a deterministic source (the tool) decides truth.** It's also literally how Claude Code works
(TodoWrite). The reframe to *verification-as-tool-use* is what makes it viable on a 4B: a 4B is
bad at computing and bad at self-grading, but decent at **call-tool→copy-result** — the one
thing we already train (our whole grounding contract). So we're pointing the existing primitive
(skill + tool + ground) at the agent's OWN progress. The audit skill is just another skill; the
calculator is just another tool.

**Verdict: worth a shot — BECAUSE the load-bearing assumption is cheap to falsify.** The whole
thing rests on one empirical unknown: *can E4B reliably run a multi-step call→read→mark loop
without losing the thread?* (small models are weak at long-horizon state). Not answerable from
theory; testable in ~a day. Cheap-to-test + high-upside = run it.

**Structural/technical calls I committed to:**
1. **Add a real compute tool — `python` (sandboxed) and/or `calc`. NOT bash** (too much
   sandbox/security surface for a local demo model, buys nothing the fabrication problem
   needs). This is the keystone and is valuable INDEPENDENT of the checklist idea — it kills
   numeric fabrication directly.
2. **The audit procedure is a SKILL.md in the catalog, not system code** — loads via
   progressive disclosure only for goals that need it; its body says "for each box, get
   evidence from a tool, mark from the output, never assert." Stays inside the contract we
   already train.
3. **Split determinism honestly.** Tool-checkable boxes ("the math is right", "the plan says
   X", "N matches in the file") are genuinely grounded by the executor — that's most of what
   our harness needs. Semantic boxes ("the advice is good", "I understood intent") **stay
   soft — do NOT let the model tool-audit them** (no oracle → audit theater). Tools extend how
   far determinism reaches; they don't close it.
4. **A complexity router decides whether to plan+audit at all.** Most turns ("best move
   here?") must NOT trigger a checklist (slow/absurd on the local 4060). Simple → existing
   fast path; compound/verifiable → plan path. This is a 4th behavior alongside
   fast/think/auto: **`plan`**.
5. **Hard loop cap + honest-partial abort.** Non-negotiable — an unterminating verify-loop
   hangs a local model. Cap → emit "did 2 of 3, blocked on X."
6. **Seq is a go/no-go gate, measured before training.** goal block + checklist + audit-skill
   body + audit tool calls + their results could blow 1664. Measure the rendered length of a
   realistic 3-box audited chain FIRST. If it doesn't fit, the architecture forces
   E4B-at-higher-seq (more VRAM, slower) — a real cost.

**The insight that ties it together:** this **subsumes the multi-skill gap.** A compound goal
naturally decomposes into a multi-box checklist, each box binding a different skill. So instead
of a bespoke "multi-skill slice," the principled version is: train the model to decompose any
goal into a checklist; compound goals yield multi-box, multi-skill plans. Same training, more
general capability.

### Does this overhaul the existing data? NO — additive.
The complexity router is what makes it additive (the same way fast/think/auto are signal-gated).
Existing 75k rows stay valid as the **no-plan majority**. What gets touched, all lightly:
- new tool(s) in the catalog — `tool_manifest` is per-row, so existing rows just don't list it.
- new slices (compute-grounding, compound-plan, audited-plan) — additive, like V1_Q last week.
- contract text renders plan-mode instructions **conditionally** (same mechanism as
  `_render_reasoning(mode)`) — non-plan rows render byte-identical to now.
- mix rebalanced (~70% existing / ~30% plan+audit), rows reused.
The ONLY scenario that forces a *targeted* regen (not overhaul): if Stage 0 shows compute-
grounding should live everywhere numbers appear → re-roll the ~4 chess-eval slices (D/E/G/I) to
ground eval via the executor. Cheap (full regen is minutes); contract + other 24 slices unmoved.

### The staged experiment (cheapest-first, each a go/no-go)
- **Stage 0 — test the assumption (≈1 day, independent value):** add `python`/`calc` tool +
  a few hundred SFT rows where the model grounds a numeric claim by CALLING it instead of
  asserting. Train, serve, measure: does E4B reliably call→read→narrate the computed value?
  **If no → STOP, the architecture can't stand.** If yes, we've shipped a fabrication fix worth
  having on its own.
- **Stage 1 — checklist on compound goals only:** self-authored `<goal>`+`<plan>`, system
  tracks structural boxes, model works each (this IS the multi-skill gap, generalized). Measure
  seq here. Train decompose→work→synthesize.
- **Stage 2 — the audit skill:** load it for complex goals, verify the tool-checkable boxes via
  the executor, loop-cap + abort. Only if Stages 0–1 hold.

### Open creative fork (user decides, not technical)
**How much does the user SEE?** Transparent agent (checklist + box-ticks visible in chat, like
Claude's todos — builds trust, costs tokens) vs. internal scaffold with only natural narration
surfaced (cleaner UX, harder to debug). Changes the SFT targets; decide before Stage 1.
Stage 0 doesn't depend on it.

---

## 3. Hardware / Colab decision (asked + answered)

**Don't buy Colab Pro yet.** Reasoning:
- A T4 is a T4 — Pro ($10/mo) doesn't give a better E4B QLoRA run; it buys **reliability +
  background execution** (a velocity buy, not capability).
- We already have the strong FREE option wired: **Kaggle** (`kaggle_e4b_qlora.ipynb`), 30h/week,
  12h/session, more stable than free Colab. It carries E4B-single-GPU-seq-1664.
- The new architecture doesn't change the training GPU need (Stage 0 is a small SFT add; same
  QLoRA recipe). The bottleneck is **iteration count** (train→eval→adjust ×3), where free-tier
  disconnects bite — not "can't fit the job."

**Run Stage 0 on Kaggle free first.** It produces the number that decides the spend (rendered
seq of an audited chain) + the go/no-go on the loop. Then:
| Stage 0 result | Decision |
|---|---|
| loop works, seq ≤ 1664 | Kaggle free carries the program; Pro is a $10 convenience, not a need |
| loop works, seq ~1800–2048+ | VRAM headroom matters → **Colab Pro+ / L4 or A100 (~$50)** justified — on the measured number |
| iteration friction killing you (disconnects) | **$10 Colab Pro** as a pure velocity buy, month-to-month |

**The hardware worry no Colab tier fixes:** the real constraint is the **RTX 4060 (8 GB)
serving the q4_0 GGUF**. The audit loop ADDS steps per turn (each box-check = a tool round-trip)
→ more tokens, more passes, slower wall-clock locally. Colab can't help serving. If anything
threatens "truly agentic on OUR system," it's local inference latency of a multi-step loop.
**Measure serve-side latency of a 3-box audited turn on the 4060 early.**

---

## 4. Run commands

```bash
# data: regenerate + build + gate (run generate/build from src/llm; gate from repo ROOT)
cd src/llm && PYTHONPATH=. python -m llm_dataset.v1.generate --profile v1.2
cd src/llm && PYTHONPATH=. python -m llm_dataset.v1.build --profile v1.2
python scripts/final_corpus_audit.py          # <-- from repo root; GATE: PASS required
python scripts/make_sample_doc.py             # regenerate the human-inspection doc

# tests (Windows pytest buffers + can stall; if a run hangs, kill pytest procs and re-run.
#   the dataset suite needs cwd=src/llm + PYTHONPATH=.; skip test_annotator.py which probes Stockfish)
cd src/llm && PYTHONPATH=. python -m pytest llm_dataset/v1/tests -q -p no:cacheprovider \
    --ignore=llm_dataset/v1/tests/test_annotator.py -o addopts=""     # 107 pass
python -m pytest src/llm/backend -q                                    # serve tests
python -m pytest src/llm/llm_training -q                               # trainer

# serve locally (two terminals)
npm run server            # model_server, weights on :7861 — leave running
npm run dev               # weightless app on :7860, hot-reload; http://127.0.0.1:7860
#   adapter:  npm run server -- "A:/path/to/adapter"

# train (Kaggle/Colab notebook is the real path) / export
cd src/llm && python -u -m llm_training.run_train
cd src/llm && python -u -m llm_training.export_gguf ../../runs/<run-dir>
```

**Windows gotchas observed this session:** (1) pytest backgrounds and **buffers output until
process exit** — a "stalled" run is usually just buffering; if truly hung, kill `python.exe`
pytest procs (zombies accumulate across sessions and cause lock contention) and re-run with
`-o addopts=""`. (2) `cd` inside a chained Bash command persists into later calls — the gate
must run from repo root; verify `pwd` if a relative path 404s. (3) `defaultdict`-style import
slips will crash the gate AFTER the ~3-min tokenize pass — it re-tokenizes on re-run, so get
imports right first.

---

## 5. Relevant files, docs, memory

**Corpus generator (source of truth for behavior):** `src/llm/llm_dataset/v1/`
- `contracts.py` (SLICES, RULES, MAX_TOOL_CALLS), `profiles.py` (accepted_target 75k),
  `generate.py` (DEFAULT_PLAN), `build.py` (split + final-text de-leak), `sampler.py`,
  `domains.py` (V1_O domains + scenes + CLOSERS), `validate.py` (per-row gate), `audit.py`.
- `renderer/`: `chess.py`, `universality.py`, `skill_routing.py`, `multiturn.py`,
  `thinking.py` (mode gating, `_NEVER_THINK_KINDS`), `finals.py`, `synth_engine.py` (NEW —
  seeded engine scenes), `chess_kb.py` (topic-keyed I/K), `universality_prompts.py` (FINAL_POOLS).

**Training/serving:** `src/llm/llm_training/` (`system_prompt.py` = the contract,
`data_pipeline.py` = tokenize + assistant mask + GROUND_WEIGHT fact up-weighting,
`chat_format.py` = the tool-role remap, `train_unsloth.py`, `run_train.py`, `kaggle_e4b_qlora.ipynb`,
`colab_e4b_qlora.ipynb`). Local tokenizer at `src/llm/models/gemma4_e2b`.
Backend/serve: `src/llm/backend/` (`inference.py` translates `<skill>`→canonical internally,
`tool_hints.py`, `tools.py`, `plugins/`).

**Scripts:** `scripts/final_corpus_audit.py` (the permanent gate — KEEP RUNNING IT),
`scripts/make_sample_doc.py` (human inspection doc). The `scripts/_audit_*.log` /
`_audit_out.json` / `_dup_*` are run artifacts, gitignore-able, not committed.

**Docs:** `docs/README.md` (index), `docs/harness-architecture.md`,
`docs/2026-06-13-v1.2-random-sample-inspection.md` (the sample doc),
`docs/2026-06-06-v1.2-dataset-alignment-audit.md` (Option B basis).

**Memory** (`~/.claude/projects/.../memory/`, loaded each session via MEMORY.md):
`seq-ceiling-v1g-truncation`, `sft-final-diversity`, `seq-dominated-by-harness-contract`,
`gemma-template-drops-tool-role`, `unsloth-load-anomaly-e2b`, `ddp-not-viable-e2b-t4`,
`chess-agent-skill-tool-contract`, `harness-first-versatility`,
`flexible-model-vs-deterministic-layers`, `deterministic-routing-restraint`,
`chess-sft-generalization-not-phrase-memorization`, `always-show-run-commands`.

---

## 6. Constraints (unchanged, still binding)

- Never edit/import `legacy [ignore]/` (gitignored archive bin).
- No secrets in code/logs/commits; HF token via Kaggle/Colab Secrets.
- Commit every turn; **push/PR only when the user explicitly asks** (branch
  `feat/chess-coach-sft` — NOT yet pushed; first push will not be a force-push).
- Stage intended files only (no `git add -A`); `*.gguf`/`*.safetensors` gitignored — commit
  code, not weights. The working tree has lots of untracked noise (`.codegraph/`, `build/*.log`,
  stray `scripts/*.ps1`, `src/llm/build/`) — do NOT blanket-add.
- Watch long shells; free GPU memory between heavy runs; kill stale :7860/:7861 + zombie pytest.

---

## 7. My personal notes (candid, scientist-to-next-agent)

- **The corpus is genuinely ready — don't re-litigate it.** Three independent passes (QC,
  diversity, full-corpus gate) landed it at GATE: PASS with 0/75,060 validation failures and
  the worst-repeated-final down 2778→273. If the next session is tempted to "improve naturalness"
  more, STOP: the remaining low-diversity slices are bounded by design. The lever now is
  **training + eval on real hardware**, not more data polish.
- **The biggest unknown in the whole project is empirical and unmeasured: can E4B carry a
  multi-step loop?** Everything in §2 hinges on it. Do NOT design the full Stage 1/2 machinery
  before Stage 0 answers this. I almost started building a multi-skill slice before realizing
  the checklist architecture subsumes it — resist building the special case.
- **Determinism honesty is the recurring trap.** Twice this session the bug was "the narration
  is independent of the tool result" (V1_G canned final; the whole grounding ethos). The audit
  architecture is the same risk one level up: let the model self-grade and you get theater.
  The fix is always "make a TOOL the source of truth, model copies it." Hold that line.
- **Seq 1664 is the silent killer.** ~85% of every row is the contract; finals have almost no
  room. Every feature that adds final/think content (engine scenes, closers, and especially the
  §2 goal+checklist+audit blocks) trades directly against the ceiling. ALWAYS measure rendered
  length on the REAL seed (`20260525`) with the real tokenizer — a sample seed hid the V1_G
  overflow this session. The gate now enforces this; trust it, run it.
- **The product's real bottleneck is serving, not training.** A 4060 running a q4_0 GGUF
  through a multi-step audit loop is the thing most likely to feel un-agentic (latency), and no
  cloud spend fixes it. Measure serve latency early; it may cap how deep the audit loop can go.
- **Don't expand the deterministic serve layer per-symptom.** Memory
  `deterministic-routing-restraint` / `flexible-model-vs-deterministic-layers`: mis-routes are a
  TRAINING signal, not a reason to bolt on another regex. The trained model is the flexible
  general caller; keep the deterministic layer minimal.
- **Working rhythm with this user:** they want the honest scientist read (rate things 1-10, name
  the real defect, push back), they say "finish it / no worry" and mean depth, they want the
  bash command to watch every long job, and they explicitly warned against "doing things that
  don't help." They reset sessions to keep context clean — so over-document in handoffs (this
  file) rather than assume continuity.

---

## 8. Immediate next action for the fresh session

**Stage 0 is BUILT, gated, and committed (see §1.5) — the remaining step is to TRAIN it.**
Run the E4B QLoRA notebook on Kaggle on the current `v1_2` split (train 73,049 / val 2,015,
now incl. the 990 `V1_R` python-verify rows), pull the adapter, `serve_check` train/serve
base-parity, and measure the ONE thing that gates the whole §2 program + the Colab spend:
**does E4B reliably call the `python` tool, read its stdout, and narrate THAT value instead of
fabricating?** (seq is already measured — Stage-0 chain max 1469, fits 1664.)
- If **yes** → fabrication fix shipped; proceed to Stage 1 (self-authored `<goal>`+`<plan>`,
  but mind the seq: only ~2–3 audited boxes fit 1664, see §1.5).
- If **no** → STOP; the agentic loop architecture can't stand on E4B.

If instead shipping the current corpus: it's ready — run the E4B QLoRA notebook on Kaggle, pull
the adapter, `serve_check` for train/serve base-parity before trusting the GGUF, export Q5_K_M,
serve smoke on the 4060.

Push only on explicit user OK.
