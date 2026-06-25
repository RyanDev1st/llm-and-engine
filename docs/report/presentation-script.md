# Presentation script — Gemma 4 E4B harness (~4–5 min)

Polished from your draft. Read the **bold**; the rest is staging. Slide files are in
`docs/findings/report_assets/` (numbered in this order); the beat→image map is `slide-visuals.md`.
Every number is verified against `docs/report/README.md §3` — nothing rounded that contradicts a
later slide. Timed to ~4:30; **always rehearse with a clock and cut from the middle, not the ends.**

---

### 01 · Meet the model  *(your AI-gen brand slide — prompt A)*
> "Let's jump right into our model. It's a **Gemma 4 E4B** — a small, 4-billion-parameter open model,
> **4-bit quantized**, trained on a **73K-example dataset** to work with our harness and to follow a
> set procedure when it answers.
>
> It's small. **Very** small. It's definitely not the superman — or superwoman — of models."

### 02 · It's local  *(your AI-gen brand slide — prompt B)*
> "But here's the one thing that matters: **it's local.** Your data is safe — it never leaves your
> device. It's the **right tool for the job**. No more, no less."

### 03 · How it works  →  `03-how-it-works.png`
> "So how does it work? The whole thing runs on **two verbs, one action at a time.** It can **load a
> skill** — that pulls in instructions for a task — or **call a tool** — that runs a real function and
> gives back a result. It picks one, reads what came back, and acts again — like a coding agent. And it
> only uses the skills and tools **listed in the prompt**, which change every request — so it's not
> memorizing chess, it's learning to operate *any* toolset."

### 03b · Taught to think  →  `03b-reasoning-modes.png`
> "And one small thing — even though it's a small model, **not a natural reasoner**, we trained it to
> think. Four modes: **fast** answers right away, **think** reasons before every step, **auto** reasons
> only on the hard choices — that's the everyday default — and **plan** breaks a multi-step request into
> a checklist and works through it. Same model; it picks the mode from the prompt. We trained the
> reasoning *in* — and the restraint to **not over-think**."

### 04 · How we trained it  →  `04-how-trained.png`
> "Here's how it was trained. We fine-tuned a tiny **LoRA adapter** on top — the base model stays
> frozen — on **Kaggle's free 2× T4 GPUs**, across a few accounts, about **135 GPU-hours** over a couple
> weeks. The settings that matter: **4-bit QLoRA** to fit a 4B model on a free GPU; **sequence length
> 1664**, because our longest training example is ~1,655 tokens and we don't want to cut off the
> reasoning; and we **up-weighted the harness tags ×8 in the loss**, so the model's format beats the
> base model's old habits. Then we serve it — on Colab for the live site, or locally as a GGUF."

### 05 · The data  →  `05-the-data.png`
> "This is the data mix. **~73K examples** — three-quarters general-purpose, a quarter chess; chess is
> just our demo domain. And the reasoning modes are baked into the distribution, so it learns *when* to
> think versus answer fast."

### 06 · It floors out fast  →  `06-floors-out.png`
> "Here's something useful: this is the **real training loss.** It drops fast and then **floors out** —
> within roughly the first **4% of one pass** through the data. So we **don't** train on the whole 73K.
> Why? Because **~85% of every example is the same harness contract** — it nails the format quickly, and
> generalizes the rest. Cheap to train, and it learns what matters early."

### 07 · How it runs  →  *(your live chat screenshots)*
> "Now let me show you how it actually runs." *(walk through 2–3 turns from your screenshots)*
> "As you can see — it picks the right skill or tool, calls the engine for the real evaluation, and
> **explains the result instead of inventing it.** Casual, vague prompts, and it still routes correctly."

### 08 · Does the training help?  →  `08-result-routing.png`
> "Does the fine-tune actually help? On held-out tests it never saw: routing accuracy went from
> **49.6% to 88.7%.** The base model **over-fires tools** — it grabbed a tool 55 times when it should
> have just loaded a skill; ours cut that to 7. It learned **when to act** — that's the hard part."

### 09 · Does it generalize?  →  `09-result-generalizes.png`
> "And a *different* test: can it finish a task in a domain it **never trained on**? On real cooking,
> music, wellness, and tax prompts, it **completed 91.7%** of them and grounded its answer in the tool
> result **95%** of the time." *(close)* "Small model, trained on free hardware, runs locally — and it
> generalizes past chess. That's the result."

---

## If asked (have it ready; don't volunteer)
- **"Why is the baseline only ~50%?"** It's not bad at chess — it's bad at *restraint*; it fires tools
  when it shouldn't. The fine-tune teaches it *when not to act*.
- **"Wait, 91.7% vs 88.7% — did it drop?"** No — **different tests.** 88.7% is routing on chess val;
  91.7% is task completion on unseen domains. They answer different questions.
- **"Show me the per-class data."** → `backup-confusion.png` (126/142 correct, tool F1 0.42→0.81).
- **"How fast is it?"** ~3 tokens/sec on a free T4 — compute-bound on free hardware; quantizing to
  GGUF roughly doubles it locally.
- **"Is the chat real?"** Yes — captured verbatim from the live serve loop (timing + tok/s shown).

## Timing
~4:30 at a calm pace. If you must cut: **05 → one sentence**, and merge **08+09** into one "50→89 on
routing, 92% on unseen domains" beat. **Never cut 01, 02, 03, or 07** — they carry the story.
