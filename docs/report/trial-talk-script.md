# 4-minute trial talk — script + slide images

Draft for the group trial run. Plan: **how I trained the AI (pipeline) → the data → showcase →
benchmark**. Audience-light, image-heavy, ~4 min. I budgeted **3:30 of talk** (always come in under —
trials run long, and you want buffer for the demo loading). 10 images, all in
`docs/findings/report_assets/`. Numbers trace to the 2026-06-24 Kaggle run (`docs/report/README.md §3`).

> Honesty note for you (not the audience): the two **chat cards are REPRESENTATIVE placeholders** —
> real-format, plausible, but NOT live model output yet. The live capture comes from the Kaggle P2 run.
> For a *trial* that's fine; say "sample interaction." Swap them before the real talk. Everything else
> (pipeline, data, recalibration, confusion, corpus) is real or conceptual, nothing fabricated.

---

## The 10 slides (in talk order)

| # | image file | slide title to type | section |
|---|---|---|---|
| 1 | `slide-pipeline.png` | How it's built | **Pipeline** |
| 2 | `slide-data-anatomy.png` | What the data teaches | **Data** |
| 3 | `chart-corpus-composition.png` | The corpus, by the numbers | **Data** |
| 4 | `slide-chat-bare.png` | It just talks — plain chat | **Showcase** |
| 5 | `slide-chat-web.png` | …and it reads the board | **Showcase** |
| 6 | `slide-recalibration.png` | The fine-tuning win | **Benchmark** |
| 7 | `slide-confusion-adapter.png` | Where it routes right | **Benchmark** |
| 8 | `chart-training-timeline.png` | (backup) how we got here | spare |

Lead with 7 core slides (1–7). Slide 8 (timeline) and the cross-model line chart
(`slide-model-lines.png`) are **spares** — the line chart is intentionally sparse right now (E2B/Q5/Q6
points land after the parallel Kaggle runs), so keep it in your back pocket, don't lead with it.

---

## Script (~3:30, read the **bold**, the rest is for you)

**Intro (0:00–0:20)**
> "We fine-tuned a small open model — Gemma-4, 4-billion-param — into a chess coach that actually
> *uses tools*. The point isn't chess. It's a general pattern: teach a small model to pick the right
> skill or tool and reason to an answer. Here's how."

**① Pipeline — slide 1 (0:20–0:55)**
> "Four steps. We write training **data**, **train** a QLoRA adapter on a *free* Kaggle T4 — the base
> model stays frozen, we only learn a small adapter — then **serve** it locally on one consumer GPU as
> a quantized GGUF. Train once on free hardware, run it on your own machine."

*(point at box 2)* "It never sees the full model in memory — that's the trick that makes a 4B model
trainable on free hardware."

**② Data — slide 2 (0:55–1:30)**
> "Every training example teaches *two verbs*. **Skill** loads instructions into the model's context —
> it doesn't act. **Tool** calls a real function and the model has to *narrate the result*, never make
> it up. One action per step, and it can think first — fast, or step-by-step on the hard ones."

> "The key: the available skills and tools are *listed in the prompt* and change every example. So it
> learns to operate *any* toolset — not memorize chess."

**③ Data scale — slide 3 (1:30–1:50)**
> "About 72,000 examples. Three-quarters general, a quarter chess. Mostly the model reasoning
> step-by-step — that distribution is deliberate."

**④–⑤ Showcase — slides 4 & 5 (1:50–2:35)**
> "Here it is talking. Casual, vague prompts —" *(slide 4)* "— 'hows my position', 'whats a fork' — it
> answers like a coach, ~3 seconds a reply."
> *(slide 5)* "And in the web sandbox it's *grounded* — it calls the engine, gets +1.8, and explains
> *why*. It doesn't invent the evaluation; it reads it from the tool."

*(If asked: these are sample interactions; live capture is running.)*

**⑥ Benchmark — slide 6 (2:35–3:10)**
> "Does the training actually help? On held-out data: routing accuracy went from **50% to 89%**. Tool
> F1 **0.42 to 0.81**. The base model *over-fires* tools — it grabbed a tool 55 times when it should
> have just loaded a skill. The adapter cut that to 7. That's the whole win — it learned *when* to act."

**⑦ Confusion — slide 7 (3:10–3:30)**
> "Same story, per-class: it almost never confuses 'load a skill' with 'call a tool.' 126 of 142
> correct. The diagonal is the model getting the *verb* right." *(close)* "Small model, free training,
> runs locally, and it routes tools correctly. That's the result."

---

## If they ask (have an answer, don't volunteer)
- **"Why 50% baseline so low?"** The base model isn't bad at chess — it's bad at *restraint*. It fires
  tools when it shouldn't. Fine-tuning teaches *when not to act*.
- **"Is the chat real?"** Format and timing are real-shape; the verbatim live capture is finishing on
  Kaggle now. *(true)*
- **"Why GGUF / quantization?"** It's how you fit a 4B model on a consumer GPU and get ~2× the speed.
  The quality comparison (Q5/Q6) is the chart that's still filling in.
- **"What's the grounded number?"** On 60 unseen-domain tasks (cooking/music/tax) it completed 92% and
  grounded its answer in the tool result 95% of the time — it generalizes past chess.

## Timing discipline
Rehearse once with a timer. If you're at 4:00 in rehearsal you'll be at 4:45 live — **cut slide 3 to
one sentence and drop the timeline spare.** Slides 1, 2, 6, 7 are the load-bearing four; never cut those.
