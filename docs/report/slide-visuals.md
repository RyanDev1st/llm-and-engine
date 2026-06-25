# Slide visuals — your script → the image for each beat

You write the talk; this maps each beat of your draft to the **visual** that illustrates it. Two kinds:
- **[RENDERED]** — a factual/data image I generated from real artifacts (in `docs/findings/report_assets/`).
  Regenerate any time with `python -m llm_training.report.deck`.
- **[GENERATE]** — an illustrative/brand image better made on ChatGPT/Gemini. I give you the prompt +
  the text to overlay in PowerPoint (AI image tools can't spell exact facts — put those in PPT yourself).

All numbers on the rendered images trace to real artifacts (run config, `runs/full_train.log`,
`docs/report/README.md §3`). Nothing invented.

---

## Beat → visual

Files are numbered by talk position (`NN-name.png` in `docs/findings/report_assets/`) so they sort in
order. Your slides fill 01-02 and 07.

| # | File | Your line (draft) | Visual |
|---|---|---|---|
| 01 | *(yours)* | "gemma 4 e4b, 4-bit, 73k… small, not the superman" | **[GENERATE]** — prompt A below |
| 02 | *(yours)* | "it's local. Your data never leaves your device. Right tool, no more no less" | **[GENERATE]** — prompt B below |
| 03 | `03-how-it-works.png` | "how it works — the think loop" | **[RENDERED]** REAL trace (think-mode, slice A): user→goal+think+skill→body→think+tool→data→think+answer. 8 steps from one row |
| 03b | `03b-reasoning-modes.png` | "taught to think — 4 modes" | **[RENDERED]** FAST/THINK/AUTO/PLAN cards with real extracted examples + <goal>/<think> rules |
| 04 | `04-how-trained.png` | "here's the pipeline… max seq, ranks, why" | **[RENDERED]** flow + knobs+why |
| 05 | `05-the-data.png` | "here's the distribution… the slices" | **[RENDERED]** mode mix + slices |
| 06 | `06-floors-out.png` | "floors out quick, don't need the whole dataset… why" | **[RENDERED]** REAL loss curve |
| 07 | *(yours)* | "how it runs [the chats]" | **YOU** — your Kaggle chat screenshots (real verbatim runs) |
| 08 | `08-result-comparison.png` | "does the fine-tune help?" | **[RENDERED]** grouped bars: E4B base+harness 49.6% vs v4 adapter+harness 88.7%, +39.1% delta. Tool false-positives 55→7 in caption |
| 09 | `09-result-generalizes.png` | "…and does it generalize?" | **[RENDERED]** Q2, 91.7% unseen-domain |
| — | `backup-confusion.png` | (only if asked "show the data") | **[RENDERED]** per-class matrix, the same 88.7% |

That's ~9 slides: 2 generated + 5 rendered + your chats.

**Why two benchmark slides, not three:** the old deck showed `89%` (routing) AND `88.7%` (confusion
matrix) — the SAME number twice — plus `92%` completion, which read as a drop. Now: **exact numbers
only** (88.7 / 91.7, so they match the confusion backup and don't look rounded/fudged), the redundant
`55→7` slide is folded into the routing caption, and the two results are labelled **Q1 (routing) vs
Q2 (unseen-domain completion)** so the audience can't misread 91.7 as "less than 88.7" — they answer
different questions on different test sets.

---

## The [GENERATE] prompts (ChatGPT / Gemini)

Aesthetic for all: **dark charcoal background (#11141a), warm gold (#c8a24a) accents, one cool blue
accent, lots of negative space, flat editorial vector, fine linework, premium keynote feel. No text in
the image. 16:9. Avoid: neon, glowing-brain clichés, generic cyber-padlocks, clutter, AI slop.**

### Prompt A — "Meet the model" (small, not a superman)
> Minimalist editorial illustration, 16:9. Near-black charcoal background (#11141a) with warm gold
> (#c8a24a) accents and one muted blue. A single small, friendly geometric figure (a little robot or a
> rounded chess pawn with simple eyes), modest and approachable, standing confidently but clearly
> SMALL — deliberately not a muscular superhero, no cape. Generous negative space around it. Flat
> vector, fine linework, restrained palette, sophisticated tech-keynote style. No text. Avoid neon,
> avoid glowing brains, avoid clutter.

**Overlay in PPT:** big — `Gemma 4 · E4B`; small — `4-bit · 73K examples · built for our harness`;
tagline — `Small. And right for the job.`

### Prompt B — "It's local" (privacy / on-device)
> Minimalist editorial illustration, 16:9. Near-black charcoal background (#11141a), warm gold
> (#c8a24a) accents, one muted blue. Concept: on-device privacy — a single laptop or desktop tower with
> a soft gold ring/shield around it and data flowing in a small CLOSED loop inside the device, nothing
> escaping outward. Calm, trustworthy, elegant. Lots of negative space, flat vector, fine linework. No
> text. Avoid clouds, avoid padlock clichés, avoid neon — keep it understated and premium.

**Overlay in PPT:** big — `It's local.`; sub — `Your data never leaves your device.`; tagline —
`The right tool for the job — no more, no less.`

### Prompt C — OPTIONAL hero for "taught to think" (use only if you'd rather not use the loop diagram)
> Minimalist editorial illustration, 16:9. Charcoal background (#11141a), gold (#c8a24a) accents. A
> small robot/figure with a single clean looping arrow tracing a deliberate cycle around its head
> (plan → act → check), understated and elegant. Negative space, flat vector, fine linework. No text.
> Avoid glowing-brain clichés, avoid neon.

**Overlay in PPT:** `Small — but we taught it to think.`

---

## Notes
- If you want any **[GENERATE]** beat as a plain factual card instead (e.g. you don't like the AI
  image), tell me and I'll render a fallback in the same navy/gold style.
- The rendered images use a light background (clean for PPT); the generated ones are dark — if you want
  them to match, either put the rendered charts on a dark PPT slide master, or ask me to render dark.
- Backup not in the main flow: `backup-confusion.png` (per-class proof — the same 88.7%).
- Removed as bad/misleading (don't use): the v2→v3→v4 timeline (single floating point + overlapping
  boxes + stale 96%) and the cross-model line chart (only 1 real point until the Q5/Q6/E2B Kaggle runs
  land). Both regenerate from a real multi-version GPU run via `report.version_eval` if needed later.
