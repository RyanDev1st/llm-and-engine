# 4-minute trial talk — script + slide images

Draft for the group trial. Order: **pipeline → data → showcase → benchmark**. Audience-centric:
one idea per slide, the number does the talking, minimal text. Budgeted to **~3:30** (always come in
under — trials run long). Images in `docs/findings/report_assets/`. Every number traces to the
2026-06-24 Kaggle run (`docs/report/README.md §3`) — verified, nothing invented.

> **For you, not the audience:** the two **chat cards are representative placeholders** — the *timing*
> is real (~3 tok/s on a T4, decode-bound), the *wording* is illustrative until the live Kaggle P2
> capture lands. Say "sample interaction." Everything else is real measured data or a concept diagram.

---

## The 8 core slides (in talk order)

| # | image file | one-line point | section |
|---|---|---|---|
| 1 | `slide-pipeline.png` | free GPU to train, your GPU to run | **Pipeline** |
| 2 | `slide-two-verbs.png` | skill = load know-how, tool = act | **Data** |
| 3 | `slide-scale.png` | 72K examples, 4B params, 1 free GPU | **Data** |
| 4 | `slide-chat-bare.png` | it talks like a coach (slang/vague) | **Showcase** |
| 5 | `slide-chat-web.png` | it reads the board / grounds answers | **Showcase** |
| 6 | `slide-win-routing.png` | **50% → 89%** routing | **Benchmark** |
| 7 | `slide-win-restraint.png` | **55 → 7** tool over-fires | **Benchmark** |
| 8 | `slide-generalizes.png` | **92%** on unseen domains | **Benchmark** |

**Backups (only if asked — do NOT put in the main flow):**
- `slide-confusion-adapter.png` — the per-class proof matrix (technical; the empty `none` row needs a
  word of explanation, so keep it in your pocket for a "show me the data" question).
- `chart-corpus-composition.png` — detailed slice breakdown (slice names are cryptic; backup only).
- `chart-training-timeline.png` — the v2→v3→v4 retrain history.
- `slide-model-lines.png` — cross-model chart; **sparse** until the Q5/Q6/E2B Kaggle runs finish.

---

## Script (~3:30 — read the **bold**, rest is for you)

**Intro (0:00–0:20)**
> "We fine-tuned a small open model — Gemma-4, ~4 billion params — to *use tools* reliably. The point
> isn't chess; chess is just our demo. It's a general recipe: teach a small model to pick the right
> skill or tool and reason to an answer."

**① Pipeline — slide 1 (0:20–0:50)**
> "Four steps. We write the **data**, **train** a small adapter on a *free* Kaggle GPU — the big model
> stays frozen — and **serve** it on one consumer GPU. Train free, run local."

**② The idea — slide 2 (0:50–1:20)**
> "The whole trick is two verbs. A **skill** loads know-how into the model's head. A **tool** runs a
> real function and the model explains the result — it never makes the number up. And the available
> skills and tools are *listed in the prompt and change every time* — so it learns to drive *any*
> toolset, not memorize chess."

**③ Scale — slide 3 (1:20–1:40)**
> "72,000 training examples, a 4-billion-param model, on one free GPU. Three-quarters general, a
> quarter chess. And it's taught *when* to think versus answer fast."

**④–⑤ Showcase — slides 4 & 5 (1:40–2:25)**
> *(slide 4)* "Here it is — casual, vague prompts: 'hows my position', 'whats a fork'. It answers like a
> coach."
> *(slide 5)* "In the web sandbox it's *grounded* — it calls the engine, gets +1.8, and explains *why*.
> It reads the evaluation from the tool instead of inventing it."

> *(on speed, only if it comes up — own it):* "On a free T4 it's ~3 tokens a second, so a reply is
> ~15–20 seconds. It's compute-bound on free hardware — which is exactly why the last step quantizes it
> to run faster locally."

**⑥ Routing win — slide 6 (2:25–2:50)**
> "Does the fine-tune actually help? On held-out tests it never saw, routing accuracy went from **50%
> to 89%**. That's the core result."

**⑦ Restraint — slide 7 (2:50–3:10)**
> "And it learned *restraint*. The base model grabbed a tool **55 times** when it should have just
> loaded a skill. Ours: **7**. It learned *when not to act* — that's the hard part."

**⑧ Generalizes — slide 8 (3:10–3:30)**
> "Best part: we tested it on **cooking, music, wellness, tax** — domains it *never* trained on. It
> completed **92%** of those tasks and grounded its answer 95% of the time." *(close)* "Small model,
> free training, runs locally, and it generalizes past chess. That's the result."

---

## If they ask (answers ready — don't volunteer)
- **"Why is the baseline only 50%?"** The base model isn't bad at chess — it's bad at *restraint*. It
  fires tools when it shouldn't. Fine-tuning teaches it *when not to act*.
- **"What's F1 / how do you measure routing?"** Per-class: tool-use F1 went 0.42→0.81, skill F1
  0.56→0.94. *(That's the `slide-confusion-adapter.png` backup — 126/142 correct.)*
- **"Is the chat real?"** Timing is real (~3 tok/s on a T4); the verbatim transcript is finishing on
  Kaggle now.
- **"Why quantize / GGUF?"** It's how a 4B model fits a consumer GPU and decodes ~2× faster.

## Timing discipline
Rehearse with a timer. If you hit 4:00 in rehearsal you'll run 4:45 live — **cut slide 3 to one
sentence; merge slides 6+7 into one "50→89, and over-fires 55→7" beat.** The load-bearing four are
**1, 2, 6, 8** — never cut those.
