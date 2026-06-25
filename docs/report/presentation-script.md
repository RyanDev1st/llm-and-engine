# Presentation script — Gemma 4 E4B harness (~4 min)

Read the **bold**. Each section maps to one slide. Files in `docs/findings/report_assets/`.
Every number from `docs/report/README.md §3`. ~4:00 at a calm pace. Rehearse with a clock.

---

### 01 · Meet the model  *(your AI-gen slide, prompt A)*
> "Let's jump right in. Our model is a **Gemma 4 E4B** — 4-billion parameters, **4-bit quantized**,
> trained on a **73,000-example dataset** to work with our harness and to follow a procedure when it
> answers. It's small. **Very** small. Definitely not the superman — or superwoman — of models."

### 02 · It's local  *(your AI-gen slide, prompt B)*
> "But here's the one thing that matters: **it's local.** Your data is safe — it never leaves your
> device. It's the **right tool for the job.** No more, no less."

### 03 · The thinking loop  →  `03-how-it-works.png`
> "And here's the trick — despite being a small model, we **trained it to think.** The flow runs in
> two halves: the **harness** handles the tools and the board on the left, and the **model** runs an
> inline loop on the right — it commits a goal, loads a skill, calls a tool, checks if it's done,
> and loops back if not. Eight steps, two verbs — `<skill>` loads guidance, `<tool>` calls a
> function — **one action at a time.**"

### 03b · Four reasoning modes  →  `03b-reasoning-modes.png`
> "Same model, four modes. **Fast** answers directly — no thinking. **Think** reasons before every
> step. **Auto** reasons only on the hard choices — that's our default. **Plan** breaks a complex
> request into a checklist and works through it. We trained the reasoning **in** — and the restraint
> to **not over-think.**"

### 04 · How we trained it  →  `04-how-trained.png`
> "Here's how we built it. We fine-tuned a tiny **LoRA adapter** on top — the base model stays
> frozen — on **Kaggle's free 2× T4 GPUs**, across a few accounts, about **135 GPU-hours** over a
> couple weeks. Four key settings: **4-bit QLoRA** to fit the whole model on a free GPU; **sequence
> length 1664**, because our longest training example is ~1,655 tokens and we don't cut the reasoning;
> **rank 16, all-linear** — enough capacity to learn the format without bloating past one GPU; and
> **loss-weight ×8 on the harness tags**, so the model learns the format faster than the base model's
> old output habits. Then we serve it — on Colab for the live site, or locally as a GGUF on your own
> card."

### 05 · The data  →  `05-the-data.png`
> "This is the data mix. **~73,000 examples** — three-quarters general-purpose, a quarter chess. Chess
> is just our demo domain. Each slice targets a different skill: board operations, evaluation,
> puzzle coaching, opening analysis — and the reasoning modes are baked into the distribution, so the
> model learns *when* to think versus act fast."

### 06 · It floors out fast  →  `06-floors-out.png`
> "Here's the real training loss. It drops fast and **floors out** within roughly **4% of one pass**
> through the data. So we don't train on the full 73K — it's not needed. Why? Because **~85% of every
> example is the same harness contract** — the model nails the format quickly, then generalizes the
> rest. Cheap to train, and it learns what matters early."

### 07 · How it runs  →  *(your chat screenshots)*
> "Now let me show you how it actually runs." *(walk through 2–3 turns from your screenshots.)*
> "As you can see — it picks the right skill or tool, calls the engine for real numbers, and
> **explains the result instead of inventing it.** Casual prompts, vague slang, trick questions —
> it routes correctly and stays grounded in what the tools returned."

### 08 · Does the training help?  →  `08-result-comparison.png`
> "So does the fine-tune actually help? On held-out tests it never saw: **49.6% → 88.7%.** The base
> model with just the harness was at ~50% — the harness itself carries the structure. But the adapter
> adds **39 percentage points** on top. The base model over-fired tools **55 times** when it should
> have loaded a skill; ours cut that to **7**. It learned **when to act** — that's the hard part."

### 09 · Does it generalize?  →  `09-result-generalizes.png`
> "And a different question: can it finish a task in a domain it **never trained on**? Real cooking,
> music, wellness, and tax prompts — **91.7% completed, 95% grounded** in tool results." *(close)*
> "**Small model, trained on free hardware, runs locally, and it generalizes past chess.** That's
> the result."

---

## If asked (have it ready — don't volunteer)
- **"Why only ~50% baseline?"** The base model isn't bad — it's bad at *restraint*. It fires tools
  when it shouldn't. The fine-tune teaches it *when not to act*.
- **"91.7% vs 88.7% — did it drop?"** No — **different tests.** 88.7% is routing on chess val
  (n=142). 91.7% is task completion on unseen domains (n=60). They answer different questions.
- **"Show me the per-class data."** → `backup-confusion.png`. 126/142 correct. Tool F1 0.42→0.81.
- **"How fast is it?"** ~3 tokens/sec on a free T4 — compute-bound on free hardware. Quantizing
  to GGUF roughly doubles it locally.
- **"What does grounded mean?"** The model's final answer contains the actual number from the tool
  result — it didn't invent it.

## Timing
~4:00 at a calm pace. If cut: merge **05** into one sentence, skip the **06** detail. **Never cut
01, 02, 03, or 08** — they carry the arc.
