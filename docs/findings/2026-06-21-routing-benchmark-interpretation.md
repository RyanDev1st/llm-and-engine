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

## Per-slice caveat: slice G = 0 %, H = 14 % is over-specialization, NOT a label bug

The two naming schemes in the val report are both real corpus structure: the **letter slices
(A–K)** are chess-domain rows (gold = `chess-coach`), the **V1_ slices** are the general-harness
rows. Denominators are sound.

Slices G and H are the *distractor-rich* chess rows: "I'm worried here — watch out for what?" with
`blunder-coach`, `tactic-trainer`, `endgame-drills`, `socratic-tutor` sitting in the catalog
alongside `chess-coach`. The confusion matrix shows skill-verb **recall 0.97** on these rows — the
model picks the right **verb** (a skill) but a **sibling skill name** instead of `chess-coach`, so
verb+name slice accuracy reads 0 %. Gold is defensible (`chess-coach` owns the opponent-threats
route; `blunder-coach` reviews *your own* blunders), so this is a genuine **over-specialization**
slip on hard, distractor-dense routing — not a mislabel. It is exactly why exact-name (73.9 %) sits
below verb accuracy (96.4 %).

**For the report:** lead with verb accuracy + macro-precision; cite exact-name with this caveat;
do not "fix" the denominators — they are clean.

## Transcript failure modes (real end-to-end runs on unseen domains)

From the captured `life-skills` transcript (cooking/music/wellness/tax — absent from training):

1. **Showcase win:** "convert 5 miles" → `<tool>convert_units …</tool>` → "8.05 km", clean end to
   end. Route-by-description generalises to an unseen tool.
2. **Tool-name emitted as a skill:** "set a metronome to 120 bpm" → `<skill>metronome_bpm</skill>`
   → `unknown_skill` → the model then flailed back to chess. **This is the one bug the benchmark
   actually exposed.** → fixed (below).
3. **Training-domain gravity:** on an out-of-domain failure the model defaults to `chess-coach` +
   chess moves. Mitigated indirectly by the fix above (the dead-end that triggered it is gone);
   the residual is a frozen-weights property, recorded as a known limit.
4. **Arg-extraction deflection:** "scale recipe 12 → 30" asked back instead of filling args. Frozen
   behaviour; out of scope for a harness-only fix.

## The fix this drove — symmetric verb-coercion error (deterministic, safe)

The executor already returns a corrective error when a **skill** name is called as a `<tool>`
("`is a skill, not a tool — load it with <skill>…`"). The reverse — a **tool** name loaded as a
`<skill>` — dead-ended at `unknown_skill`, which is what sent the model back to chess (failure mode
2). Added the symmetric corrective error so the loop self-corrects to `<tool>` instead:

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

- **Slice G/H over-specialization** — a routing judgment on frozen weights; a name-coercion that
  forced `chess-coach` would be too aggressive and would hurt the cases where a specialized skill is
  genuinely the better pick. Left as a documented eval caveat.
- **Arg-extraction deflection / training-domain gravity** — frozen-weights properties, not harness
  bugs. Recorded as known limits for the report's limitations section.
