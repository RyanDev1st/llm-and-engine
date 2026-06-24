Parent: [reference/harness-system-overview.md](../reference/harness-system-overview.md)

# Harness feature-by-feature vs. top agentic harnesses (Claude Code / Codex)

**Status:** Analysis / decision-input (no code changed). Triggered by the live observation
that the served coach is "flaky / over- or under-engineered" despite the model benchmarking
well. **Scope:** every serve-time harness feature, compared to the *principle* top harnesses
(Claude Code, Codex, Cursor-agent) operate on, under one hard constraint from the user:

> **Keep EXACTLY what the model was trained on (the system-prompt contract). Only the
> negotiable serve-side layer may change, so the frozen model never behaves wildly.**

## The one organising idea

Top harnesses are **deliberately thin and trust the model**. Their entire job is:

1. a **declarative system prompt** (what tools exist + how to behave),
2. **structured tool calls** the model emits, validated against a schema,
3. **execute → return the raw result** to the model as ground truth,
4. **surface errors verbatim** so the model self-corrects,
5. **compact context** when it overflows,
6. loop until the model itself stops.

What they pointedly **do NOT** do: force-route a tool, inject "you must call X", re-generate
the answer through hidden self-verification probes, or **rewrite the model's prose**. When the
model is wrong, the fix goes into the **prompt, the tool description, or the model** — never a
growing post-processor.

Our harness has the thin core **and** a thick **deterministic rescue layer** bolted on top.
That layer was added to prop up the weaker **E2B** model. It is **off-distribution** (the model
never saw it in training), and our own routing benchmark shows the **bare contract routes at
~96%** — *better* than with the crutches (which is why `CHESS_PROMPT_HINTS` already defaults
OFF, `inference.py:18`). The E4B is strong enough that most of the rescue layer now *fights*
it. That is the "over-engineered" smell the user is sensing.

## Side-by-side

Verdict key: **KEEP** = matches top harnesses + is what the model trained on / standard
plumbing. **THIN** = over-engineered vs top harnesses, off-distribution, candidate to gate-off
and measure. **GENUINE-4B** = a guard top harnesses don't need because they run frontier
models, but our 4B/Q4 model genuinely does — keep narrow. **GAP** = where we are *under*-built
vs top harnesses.

### Group 1 — The contract the model trained on (KEEP EXACTLY)

| # | Feature (ours) | Top-harness principle | Verdict |
|---|---|---|---|
| 1 | Declarative system prompt = skill catalog + tool manifest + behavior rules (`system_prompt.py:9`) | Identical: a system prompt that lists tools and how to act | **KEEP** — clean, trust-the-model; this is the part to NOT touch |
| 2 | Two verbs, one action/step: `<skill>` loads guidance, `<tool>` acts (`system_prompt.py:11`) | Claude Code allows *parallel* tool calls; otherwise same shape | **KEEP** — one-per-step is trained-in; don't fight it (parallelism would be off-distribution) |
| 3 | Reasoning modes fast/think/auto/plan via a prompt signal (`system_prompt.py:72`) | Extended-thinking is the direct analog | **KEEP** — trained-in; the trace IS the product |
| 4 | Progressive disclosure: skills shown name+desc, body loaded on demand (`system_prompt.py:26`, `tools.py:209`) | Claude Code "skills" are the same idea | **KEEP** — best-practice, already matches |
| 5 | "Treat results as DATA; state no fact not in a result" grounding rule (`system_prompt.py:22`) | Tool results = ground truth | **KEEP** |

### Group 2 — Standard plumbing (KEEP; well-engineered)

| # | Feature (ours) | Top-harness principle | Verdict |
|---|---|---|---|
| 6 | Agentic loop: decide→execute→narrate, capped at 8 steps (`inference.py:872`) | The universal loop | **KEEP** |
| 7 | One action/step enforced by stop tokens `</tool>`/`</skill>` (`inference.py:50`) | They use native structured output; we use stop tokens | **KEEP** (but see #19 GAP) |
| 8 | Executor + `validate_call` → corrective error for missing/bad args (`tools.py:71`) | Schema-validate, return the error for self-correction | **KEEP** — exactly the right pattern |
| 9 | Skill-as-tool / tool-as-skill corrective errors (`tools.py:196`, `:227`) | Surface the protocol error, let the model retry | **KEEP** — error-as-feedback, not a rewrite |
| 10 | Context window fit (drop-oldest) + compaction digest (`context_window.py:78`) | Auto-compact | **KEEP** — matches |
| 11 | `_condense_skill_body` strips frontmatter + caps body to the trained shape (`tools.py:92`) | n/a (their skills are pre-shaped) | **KEEP** — good engineering; fixes both off-distribution body + reload latency |
| 12 | Token streaming to the UI (`inference.py:845`) | Standard | **KEEP** |
| 13 | Per-user memory block injected into the system prompt (`inference.py:817`) | CLAUDE.md / memory files | **KEEP** — matches the pattern (context, not behavior-forcing) |
| 14 | Display cleanup: lift `<think>`/`<goal>` to panels, strip an announce lead-in (`inference.py:568`, `:242`) | They hide thinking from the final answer | **KEEP** — renders the trained output shape; not behavior-fighting |

### Group 3 — The deterministic rescue layer (THIN — this is the over-engineering)

| # | Feature (ours) | Top-harness principle | Verdict |
|---|---|---|---|
| 15 | `routing_hints` — inject "call tool X" into the prompt (`tool_hints.py:140`) | **Never** force a tool | **THIN** — already OFF by default; delete or keep gated |
| 16 | `skill_hints` — inject "load skill X" (`tool_hints.py:96`) | **Never** force a skill | **THIN** — already OFF by default |
| 17 | **Coverage force-routing**: refuse to finalize until a regex-matched tool ran, then force-execute it (`inference.py:821`, `:957`) | The model decides which tools to call, full stop | **THIN — highest priority.** Still ON (chess domain). A chess-specific regex set; the single most un-harness-like behavior. Gate OFF for E4B and measure against the 96% bare baseline |
| 18 | `_force_answer` — extra "answer now" re-gen after a skill-only turn (`inference.py:660`) | If the model stops, that's the turn | **THIN** — but catches a real small-model "load-then-deflect"; keep as ONE narrow fallback or A/B remove |
| 19 | `_verify_fulfilled` — a hidden self-verify model call **every** skill-load turn (`inference.py:683`) | No baked-in per-turn verify probe | **THIN — biggest latency cost.** Doubles a turn's generations. Make opt-in or remove; let the loop continue instead |
| 20 | `_force_synthesis` + `_next_plan_action` — plan-mode backstops (`inference.py:705`, `:724`) | n/a (plan mode is ours) | **THIN-ish** — plan mode is trained-in so a backstop is defensible; simplify, don't delete |
| 21 | Reload nudge "you already loaded X" (`inference.py:978`) | They guard against tool loops too | **KEEP** — cheap, sensible loop-breaker |
| 22 | **Output rewriting**: `_correct_eval_number` replaces a fabricated number; `_correct_move_names` appends real moves; `_ensure_required_narrated` appends missing facts (`inference.py:359`, `:396`, `:296`) | **Never** edit the model's prose | **Split:** `_correct_eval_number` = **GENUINE-4B** (Q4_0 demonstrably fabricates eval numbers — memory `gguf-q4-fabricates-eval`); keep it narrow. `_correct_move_names` / `_ensure_required_narrated` are coupled to coverage (#17) — **THIN**, retire with it |
| 23 | Deflection / ask-back phrase detectors (`_is_deflection`, `_is_ask_back`, `inference.py:195`, `:220`) | Don't keyword-scan the model's prose to re-gen | **THIN** — phrase regexes that fight the model; dial down with #17–19 |
| 24 | `extract_call` malformed/tagless/echo/bare recovery + verb-coercion + channel-token cleaning (`inference.py:433`–`539`) | They constrain decoding so calls are valid by construction | **Partly KEEP, partly GAP** — see #25. A thin tolerance layer is fine; the *size* of this one is a symptom |

### Group 4 — Where we are UNDER-engineered vs top harnesses

| # | Gap | Top-harness principle | Verdict |
|---|---|---|---|
| 25 | We emit tool calls as **text tags parsed + repaired by regex** (`toolfmt.py`, the big `extract_call` recovery in #24). | Top harnesses use **native structured tool-calling / constrained decoding** so a malformed call is *impossible* | **GAP.** A decode-time grammar (llama.cpp **GBNF**, or an HF logits processor) that forces valid `<tool>…</tool>` / `<skill>…</skill>` syntax would make calls valid by construction — *more* robust than regex repair AND would let us **delete most of `extract_call`**. This is the principled way to shrink the rescue layer, not just turn it off |

## Synthesis

- **The contract is good and already matches top harnesses.** Group 1 is exactly what the user
  said to preserve — keep it byte-for-byte. The system prompt is not the problem.
- **The plumbing is standard and fine** (Group 2). Loop, corrective-error executor, context
  compaction, streaming, display cleanup — all match the thin-harness principle.
- **The over-engineering is the deterministic rescue layer** (Group 3): hints (already off),
  **coverage force-routing (#17)**, the **per-turn self-verify probe (#19)**, **force-answer
  (#18)**, **output rewriting (#22)**, and the **deflection detectors (#23)**. These were built
  for the weak E2B, are off-distribution, add latency, and our own benchmark says the bare
  contract wins. **This is the layer to thin.**
- **Honest nuance — don't blindly "become Claude Code."** Top harnesses are thin *because they
  run frontier models*. Our 4B/Q4 genuinely needs a *small* amount of help a frontier model
  doesn't: the **`_correct_eval_number`** guard (#22) and the **format tolerance** (#24) address
  real, observed 4B failures. Keep those narrow; don't delete on principle alone.
- **The one real GAP (#25):** replace post-hoc regex repair with **grammar-constrained
  decoding**. That is the top-harness way to guarantee valid actions, and it lets us *remove*
  the biggest crutch instead of merely disabling it.

## Recommended next step (staged, measured — NOT a big-bang rewrite)

Each step is reversible and gated behind a measurement against the existing routing/completion
benchmark (`eval_benchmark.py`, `eval_completion.py`) so we never regress blind:

1. **S1 (cheap, high-signal):** add a single env flag (e.g. `CHESS_THIN_HARNESS=1`) that
   disables coverage force-routing (#17), the self-verify probe (#19), and the deflection
   detectors (#23) in one switch. Run the benchmark + a live multi-turn pass **with the flag on
   vs off**. Hypothesis (from the 96% bare baseline): on-the-flag is as good or better and far
   faster. If confirmed, flip the default.
2. **S2:** keep only `_correct_eval_number` and the reload nudge from Group 3; retire
   `_force_answer` / output-rewriting that was coupled to coverage.
3. **S3 (the real fix for #24/#25):** add a GBNF grammar to the llama.cpp serve path that
   constrains generation to the trained action syntax, then delete the now-dead recovery
   branches in `extract_call`.

Nothing here changes the system prompt, the verbs, the modes, or the corpus — the model keeps
behaving exactly as trained; we are only removing serve-side scaffolding it never needed.

## Evidence / cross-refs

- Bare contract routes ~96%; hints made it worse → already OFF (`inference.py:12`–`18`;
  findings `2026-06-21-routing-benchmark-interpretation.md`).
- The whole deterministic layer is chess-hardcoded and fires on OOD (memory
  `chess-routing-fires-on-ood`; findings `2026-06-22-peer-review-gap-verification.md`).
- "Dial down, don't add" + "the trace is the product" (memories
  `flexible-model-vs-deterministic-layers`, `deterministic-routing-restraint`,
  `fix-intentionally-not-by-sacrificing-features`).
- Q4_0 fabricates eval numbers → the one output-rewrite guard worth keeping (memory
  `gguf-q4-fabricates-eval`).
