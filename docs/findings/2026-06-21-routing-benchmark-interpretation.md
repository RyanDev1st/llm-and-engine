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

## Transcript failure modes (from the captured `life-skills` transcript)

Source caveat: the transcript was generated on Kaggle; only the PNGs were synced locally, so the
items below are from the transcript as read during the Kaggle run, **not re-verifiable from local
artifacts right now**. The STRESS miss-log (above) is what will confirm items 2–4 on the next run.

1. **Showcase win:** "convert 5 miles" → `<tool>convert_units …</tool>` → "8.05 km", clean end to
   end. Route-by-description generalises to an unseen tool.
2. **Tool-name emitted as a skill (the (a) trigger):** "set a metronome to 120 bpm" reportedly
   produced `<skill>metronome_bpm</skill>` → `unknown_skill` → fallback. If real, this is the one
   bug the harness can fix deterministically (below). The STRESS miss-log will show whether it
   actually fires (target `skill:metronome_bpm` under a `wrong-verb→skill` kind).
3. **Training-domain gravity:** on an out-of-domain failure the model is reported to default to
   `chess-coach` + chess moves — a frozen-weights property if confirmed, recorded as a known limit.
4. **Arg-extraction deflection:** "scale recipe 12 → 30" reportedly asked back instead of filling
   args. Frozen behaviour; out of scope for a harness-only fix.

## The fix — symmetric corrective error (justified by code asymmetry, not just the transcript)

The primary justification is a **verifiable** asymmetry in the executor (read directly in
`tools.py`), independent of the transcript: a **skill** name called as a `<tool>` returned a helpful
corrective error ("`is a skill, not a tool — load it with <skill>…`"), but the reverse — a **tool**
name loaded as a `<skill>` — dead-ended at `unknown_skill` with no hint of the right verb. That
asymmetry is a latent correctness gap regardless of how often the model triggers it. The transcript
(failure mode 2) is the *reported* trigger; the STRESS miss-log will confirm its real frequency.
Added the symmetric corrective error so the loop self-corrects to `<tool>`:

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
