# Research brief — LLMs for tool calling into an AI engine

## Goal

Identify **training, inference, evaluation, and system patterns** that improve how LLMs **select, format, and adhere to** calls to external tools (functions, APIs, game/engine backends), including **error handling**, **multi-turn tool loops**, and **routing** between direct answers and tool use.

## Scope (in)

- Supervised / preference / RL-style methods for tool use.
- Benchmarks and metrics (e.g. agentic, API, structured output).
- Model scale effects, constrained decoding, grammar-guided generation.
- “Mode” discipline (e.g. post-tool narration only) and validation strategies.
- Production patterns: timeouts, parallel tools, idempotency, observability.

## Scope (out) for this 7-hour sprint

- Writing or modifying training/inference code in a product repo.
- Dataset generation runs (e.g. JSONL builders) — *may be researched as methodology only*.

## Product anchor (optional but useful)

**File:** `a:\Download\chess_assistant_sft_dataset_spec_v3.md`  

**Why it matters:** Defines a **small tool set**, **strict turn grammar**, **replay validation**, and **Mode 1 vs Mode 2** behavior after tool results — a concrete template for “LLM as router + narrator over an engine.”

## Research questions (prioritized)

1. What **post-2024** surveys or benchmarks best track **realistic multi-tool** and **long-horizon** tool use?
2. Which **training recipes** most improve **format adherence** vs **semantic tool choice**?
3. How do teams **validate** tool-calling datasets (static checks, execution, model-based judges) without training on judge errors?
4. What failure modes dominate at **small** (e.g. 3B–8B) vs **large** scales?
5. How do **unified prompts** vs **router + worker** architectures trade off for reliability?

## Deliverable

- Populated `findings/` and `literature/NOTES.md`.
- A short merged narrative in `SYNTHESIS.md` listing: **consensus**, **contradictions**, **gaps**, **recommended next experiments** (research/design only).

## Time box

**Window:** ~7 hours from project kickoff, research-only.  
**Started (fill in):** 2026-05-13 __:__  

## Autoresearch alignment (this sprint)

Use **$autoresearch probe** / **$autoresearch reason** / **$autoresearch scenario** *only as methodological inspiration* for questioning and debate — outputs land in `findings/`, not in code loops. Primary loop here is **read → note → cite**.
