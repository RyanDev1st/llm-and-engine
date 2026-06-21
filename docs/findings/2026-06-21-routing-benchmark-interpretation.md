Parent: docs/reference/harness-architecture.md

# Routing benchmark — interpretation & one fix it drove (Gemma 4 E4B, v4 adapter)

**Status:** complete — numbers verified against the raw Kaggle artifacts; one harness fix landed from it.
**Scope:** reads the 3-condition routing ablation (val + held-out STRESS) and the real agent
transcript, separates real model behaviour from eval-label artifacts, and records the single
deterministic harness fix the evidence justified.

## The conditions (what each isolates)

| condition | weights | harness contract | isolates |
|---|---|---|---|
| **A. adapter + harness** | trained v4 LoRA | yes | the shipped product |
| **B. base + harness** | base (LoRA off) | yes | what the **SFT weights** bought |
| **C. base, no harness** | base (LoRA off) | **no** | what the **harness contract** bought |

B and C reuse the same loaded model with the LoRA disabled — no second load, no fabricated
external baseline. Everything is reproducible from our own artifacts.

## Headline numbers

**Validation (n = 692 stratified held-out rows):**

| metric | A adapter+harness | B base+harness | C base no-harness |
|---|---|---|---|
| verb accuracy (skill/tool/none) | **96.4 %** | 82.9 % | 3.0 % |
| macro-precision | **78.3 %** | 46.2 % | 1.0 % |
| exact-name | **73.9 %** | 17.6 % | 0.0 % |

**Held-out STRESS (n = 20: messy/slang phrasing + unseen out-of-domain catalog + decline):**
A = 90 %, B = 95 % (a 1-row swing — not significant at n=20), C = 25 %.

### Reading it

- **The harness contract is doing most of the routing work**, and the SFT weights make it
  *reliable*. C → B (add the contract) lifts verb accuracy 3 % → 82.9 %; B → A (add the trained
  weights) lifts it to 96.4 % and **roughly quadruples** macro-precision (46 → 78) and exact-name
  (18 → 74). The product is the *combination*; neither layer alone is enough.
- **Verb accuracy is the honest headline (96.4 %)**, not exact-name. The classes are imbalanced
  (≈646 skill / 25 tool / 21 none in val), so macro-precision and the per-slice breakdown are the
  conservative cuts to quote, not raw accuracy.
- **C collapses to "none" on nearly everything** — the base model with no contract does not emit
  the `<skill>`/`<tool>` verbs at all. This is the cleanest evidence that the protocol is taught,
  not latent.

## Per-slice caveat: slice G = 0 %, H = 14 % — failure mode UNCONFIRMED (eval discarded predictions)

The two naming schemes in the val report are both real corpus structure: the **letter slices
(A–K)** are chess-domain rows (gold = `chess-coach`), the **V1_ slices** are the general-harness
rows. Denominators are sound — that part is verified.

What is **NOT** verified is *why* G/H score so low, because `eval_benchmark._bench` (and
`eval_confusion.evaluate`) computed the prediction `(verb, name)` and then **threw it away** —
only pass/fail counts were kept. So the raw artifacts cannot say what the model emitted. Two
failure modes fit the data and they are very different:

- **Over-specialization (wrong-name):** right verb (skill) but a sibling skill — `blunder-coach`
  / `tactic-trainer` (both confirmed present in the G/H catalogs) instead of `chess-coach`. Would
  be a harsh-but-fair miss, not a bug.
- **Wrong-verb:** the model emits the wrong KIND of action. **Slice H's prompt is literally
  "I'm worried here — undo that"**, and `undo` is a *tool* — so a verb-miss to `<tool>undo</tool>`
  is at least as plausible as over-specialization here.

The confusion matrix's skill-recall 0.97 is an **aggregate over all ~646 skill-gold rows**, NOT
slice G/H specifically, so it does not adjudicate this. An earlier draft of this finding asserted
"over-specialization" as fact; that was an inference beyond the data and has been corrected here.

**Resolution (built, not run):** `bench_misses.py` now records every missed row and the benchmark
report renders a per-slice **MISS analysis** table (wrong-name vs wrong-verb→X, plus the top wrong
target) and writes `*-routing-benchmark-misses.jsonl`. The next Kaggle run of the existing notebook
produces ground truth for G/H — and, on the STRESS suite, shows whether `<skill>metronome_bpm</skill>`
(the asymmetry fix below) actually fires. **Is G/H a problem to fix?** Unknown until that run: if it
is over-specialization, it is a documented eval caveat, not a fix; if it is a wrong-verb slip on
surface cues like "undo", that is a real routing weakness worth addressing.

**For the report (current state):** lead with verb accuracy + macro-precision; cite exact-name and
state plainly that the per-slice failure mode is pending the miss-analysis re-run. Do not "fix" the
denominators — they are clean.

## Transcript failure modes (CONFIRMED — quoted verbatim from the captured `life-skills` transcript)

The transcript is the life-skills STRESS prompts only (cooking/music/wellness/tax). The quotes
below are verbatim from the captured run (recovered from the session log). Note: the transcript
contains **no chess slice G/H rows**, so it says nothing about the (b) per-slice question — only
the val miss-log re-run can.

1. **Showcase win:** `convert: 5 miles = 8.047 kilometers` → **Coach:** "5 miles is about 8.05
   kilometers." Route-by-description generalises to an unseen tool, grounded end to end.
2. **Tool-name emitted as a skill (the (a) trigger) — CONFIRMED:**
   `set a metronome to 120 bpm` → `<skill>metronome_bpm</skill>` → `error: unknown_skill` →
   `<skill>chess-coach</skill>` (flailed back to chess). And `stressed … breathe` →
   `<skill>breathing_timer</skill>` → `error: unknown_skill` → **retried the same `<skill>
   breathing_timer</skill>`** → `error: duplicate_tool_call`. Both are tools, emitted with the
   skill verb. This is the exact bug the fix below targets — and the breathing case shows the fix
   does double duty: nothing told the model "that's a tool", so it re-tried the wrong verb.
3. **Training-domain gravity — CONFIRMED:** on the metronome OOD failure the model defaulted to
   `<skill>chess-coach</skill>`. A frozen-weights property; the fix removes the dead-end that
   triggered it, but the gravity itself is a known limit.
4. **Arg-extraction deflection — CONFIRMED:** `wanna make like 3x the cookies` loaded
   `recipe-scaler`, called `scale_recipe from_servings=1 to_servings=3`, then asked "Should I try
   to scale it to 5 servings, or is 3x okay?" — a defensible clarify, frozen behaviour, out of
   scope for a harness-only fix.

## The fix — symmetric corrective error (justified by code asymmetry, not just the transcript)

Two independent justifications, both verified: (1) a code asymmetry in `tools.py` — a **skill**
called as a `<tool>` returned a helpful corrective error, but a **tool** loaded as a `<skill>`
dead-ended at `unknown_skill` with no hint of the right verb; (2) the transcript (failure mode 2)
**confirms the model actually does this** — `metronome_bpm` and `breathing_timer` were both emitted
as `<skill>`, and in the breathing case the model re-tried the same wrong verb (→ `duplicate_tool_
call`) because nothing redirected it. Added the symmetric corrective error so the loop self-corrects
to `<tool>`:

- `backend/tools.py` — `_load_skill` now checks `_known_tool_names()` (official + compute + enabled
  plugins) before giving up; a tool-as-skill returns
  `error: '<name>' is a tool, not a skill — call it with <tool><name> ...</tool>`.
- `backend/inference.py` — user-facing narration maps it to "Let me call the right tool and try
  again." (mirrors the existing skill-as-tool narration).

Chosen as a **corrective error**, not silent auto-coercion: it matches the existing pattern, the
model is trained to recover from corrective tool errors, and it never runs a tool with missing args
behind the user's back. Covered by `backend/test_life_skills.py::test_tool_loaded_as_skill_gets_
corrective_error` (a genuinely unknown name still reports `unknown_skill` — no false coercion).

## What we did NOT change (and why)

- **Slice G/H** — not touched because the failure mode is not yet known (see the caveat above);
  fixing before the miss-analysis re-run would be guessing. If it turns out to be over-specialization,
  a name-coercion forcing `chess-coach` would be too aggressive anyway; if it is a wrong-verb slip,
  that is the thing to address — decide with data, not now.
- **Arg-extraction deflection / training-domain gravity** — frozen-weights properties, not harness
  bugs. Recorded as known limits for the report's limitations section.
