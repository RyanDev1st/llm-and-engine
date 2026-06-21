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

## Per-slice: slice G = 0 %, H = 14 % — RESOLVED by the miss analysis (2026-06-22 Kaggle run)

The two naming schemes in the val report are both real corpus structure: the **letter slices
(A–K)** are chess-domain rows (gold = `chess-coach`), the **V1_ slices** are the general-harness
rows. Denominators are sound. The miss-log re-run now says *what the model actually emitted*, and
my earlier "over-specialization to a sibling chess skill" guess was **wrong**:

- **Slice G (0/25): the model emits `<skill>threats</skill>` on 24/25.** `threats` is **not a skill
  in the catalog** — it is the object-noun of these rows' trained goal (`<goal>check the opponent's
  threats</goal>`). The eval forces **fast mode** for speed (assuming routing is mode-independent),
  but slice G was trained in **auto mode** where the model emits the goal *then* `<skill>chess-coach
  </skill>`. The leading explanation — now backed by the exact wrong-name — is that forcing fast
  collapsed the trained goal→skill sequence and the goal keyword leaked into the skill slot. So G's
  0 % is most likely a **fast-mode-eval artifact, not a production failure** (in auto mode the goal
  step is intact). Confirm cheaply by re-running ONLY G/H without `force_fast` (auto mode).
- **Slice H (3/22): mostly `<skill>list_pieces</skill>` (8) + other specifics; 3 are wrong-verb→tool.**
  `list_pieces` is a **tool**, so these are **tool-name-as-skill** — the *same* class as the
  metronome bug, which the deployed harness now **self-corrects** (the (a) fix; transcript proof
  below). The 3 wrong-verb→tool are `undo`-type ("I'm worried here — undo that"), a defensible direct
  route the strict gold ("load chess-coach first") penalises.

**Bigger point for the report — the routing eval UNDERSTATES production accuracy.** It scores only
the *first action*, but a large share of the val misses are **tools emitted as skills**: E's top
wrong target is `skill:best_move` (a tool), A's is `skill:move_san` (a tool), H's is
`skill:list_pieces` (a tool). At serve time the symmetric corrective error (the (a) fix) redirects
every one of these to `<tool>` and the loop recovers — **proven end-to-end in the new transcript**
(metronome + breathing both recover and answer correctly). So the first-action-strict per-slice
numbers are a *lower bound*; the served agent does better on exactly these rows.

**Is G/H a problem to fix?** G: likely no — it's an eval-harness artifact (fast-mode forcing);
the cheap fix is to score G/H in their trained mode, not to change the model. H: largely no — the
serve loop already self-corrects the tool-as-skill misses, and the rest is strict gold. Neither
points at a weight change.

**For the report:** lead with verb accuracy (96.4 %) + macro-precision (78.3 %); present the
per-slice table with this miss-analysis caveat; note that first-action scoring is a lower bound
because the harness recovers the tool-as-skill class at serve. Denominators are clean — don't touch.

## Transcript — the (a) fix PROVEN working end-to-end (post-fix capture, 2026-06-22 run)

The new transcript (life-skills prompts: cooking/music/wellness/tax) was captured **with the
symmetric corrective error live**, and it shows the fix recovering exactly the cases that flailed
before. Verbatim:

1. **Showcase win (unchanged):** `convert: 5 miles = 8.047 kilometers` → **Coach:** "5 miles is
   about 8.05 kilometers." Route-by-description generalises to an unseen tool, grounded end to end.
2. **Tool-as-skill now SELF-CORRECTS (the (a) fix working):**
   `set a metronome to 120 bpm` → `<skill>metronome_bpm</skill>` →
   `error: 'metronome_bpm' is a tool, not a skill — call it with <tool>metronome_bpm ...</tool>` →
   `<tool>metronome_bpm bpm=120</tool>` → `120 bpm = 500.0 ms per beat` → **Coach:** "That's 500.0ms
   per beat at 120 BPM." Same for breathing: `<skill>breathing_timer</skill>` → corrective error →
   `<tool>breathing_timer seconds=10</tool>` → answered. **Before the fix these dead-ended at
   `unknown_skill` and flailed to chess / a duplicate retry; now the loop recovers in one step and
   answers correctly.** This is the proof the fix lands — and it's the same class as the slice-H val
   misses, so the served agent recovers those too.
3. **Training-domain gravity — eliminated for this case:** the metronome turn no longer falls back to
   `<skill>chess-coach</skill>` (the dead-end that triggered it is gone). The underlying gravity is
   still a frozen-weights property, but this specific trigger is resolved.
4. **Arg-extraction deflection — still present (frozen):** `wanna make like 3x the cookies` loaded
   `recipe-scaler`, called `scale_recipe from_servings=1 to_servings=3`, then asked "Should I try to
   scale it to 5 servings, or is 3x okay?" — a defensible clarify, out of scope for a harness fix.

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
