Parent: ../implementation.md

# Context-Window System (session memory manager)

## Status

Implemented and unit-tested. Serving-integrated (`CoachLoop`), backend-accurate
token counting (HF + GGUF), live UI meter, and a deterministic eviction policy.
No model summarization (deliberately out of scope — see Next). Retrain-independent:
this is a serving/runtime change and does not touch the SFT corpus.

## Scope

### Problem

A transformer attends to a fixed token budget (`n_ctx`). The served prompt is
`system + conversation history + new user turn`. Left unbounded, a long chat
overflows the window: the backend silently drops the oldest tokens (the model
"forgets" mid-session and contradicts itself) or errors out. Before this change
the conversation history was an **unbounded list** appended verbatim every turn —
no counting, no ceiling, no eviction. Reliability hazard for a live demo.

### Design

A bounded **sliding-window memory manager** with three properties:

1. **Token-accurate budgeting.** The real tokenizer drives the math — HF
   `tok.encode`, llama.cpp `tokenize` — injected as a `count(text)->int` function
   so the policy module stays pure and backend-agnostic (chars/4 fallback for
   fakes/tests).
2. **Oldest-first eviction, recency preserved.** Keep a contiguous suffix of the
   most-recent turns that fits; evict the oldest. The system prompt and the
   current user turn are always retained; a kept suffix that would begin on an
   assistant turn is trimmed to begin on a user turn (no split pairs).
3. **Ephemeral thinking.** Tool-call/observation turns are used in-turn to write
   the reply, then dropped — only `{user, final reply}` persist to history. The
   reasoning scratchpad never accumulates across turns. (Implemented earlier this
   session; the budget below assumes it.)

### Budget formula

```
budget        = n_ctx - reserve_output - reserve_thinking - safety_margin
history_avail = budget - tokens(system) - tokens(user)
```

Defaults: `reserve_output=192` (the reply), `reserve_thinking=1024` (in-turn tool
calls + results, bounded by `MAX_TOOL_CALLS=6`), `safety_margin=64` (chat-template
framing). `n_ctx` is read from the backend (`context_limit()`): GGUF reports its
loaded `n_ctx` (default raised 2048 → 4096), HF reports `min(max_position_embeddings, 8192)`.
Reserving output + thinking up front guarantees the in-turn prompt — which grows
as tools are called — cannot exceed `n_ctx` at generation time.

### Failure modes addressed

| Failure | Before | After |
| --- | --- | --- |
| History exceeds `n_ctx` | silent truncation / crash | oldest turns evicted, never overflows |
| Per-turn thinking blows the window | unbounded | `reserve_thinking` carve-out + 6-call cap |
| System + user alone too large | crash | returned anyway, `overflow=true` flagged (degrade, don't crash) |
| Slow inference from bloated prompt | grows forever | bounded suffix |

### Modules

| Path | Role |
| --- | --- |
| `src/llm/backend/context_window.py` | `ContextWindow.fit()`, `WindowConfig`, `WindowStats`, `estimate_tokens` (pure) |
| `src/llm/backend/inference.py` | `_build_window()` wires backend tokenizer; `CoachLoop.respond` applies `fit`, returns stats |
| `src/llm/backend/model_hf.py` / `model_gguf.py` | `count_tokens()` + `context_limit()` |
| `src/llm/backend/web_app.py` | passes stats through `chat()` |
| `gemma_chat_site/static/app.js` + `styles.css` | live memory meter (used/budget, % , turns kept/evicted) |

### Observability (demo + paper)

Every reply returns `WindowStats`: `n_ctx`, `budget`, `system/history/user/used`
tokens, `turns_total/kept/evicted`, `overflow`. The UI renders a per-reply meter;
the numbers are directly quotable as a figure/table in the report.

## Evidence

```
# unit suite (pure policy invariants)
PYTHONPATH=src/llm python -m pytest src/llm/backend/test_context_window.py -q
  -> 6 passed

# serving integration (scripted model drives the real CoachLoop)
PYTHONPATH=src/llm python -m pytest src/llm/backend/test_serve_smoke.py -q
  -> 2 passed  (asserts context stats present + used_tokens <= budget)

# dataset suite unaffected by the serving change
PYTHONPATH=src/llm python -m pytest src/llm/llm_dataset/v1/tests -q
  -> 97 passed  (includes the test_leadin_shape fake fix)
```

Invariants verified by the unit suite: kept prompt token-sum never exceeds
budget; eviction is oldest-first (suffix); system + user always retained;
`overflow` flagged when they alone exceed budget; policy is tokenizer-swappable.

## Next

1. **Optional rolling summary of evicted turns.** Deterministic compaction first
   (e.g. retain a one-line factual marker); an LLM summary is deferred — a
   hallucinating summarizer would contradict the project's anti-fabrication goal.
2. **Token-aware skill-catalog trimming.** The system prompt is rebuilt every
   turn and grows per registered skill/plugin; consider budgeting the catalog
   itself when many demo skills are added.
3. **Persisted per-turn stats** to a log for a report time-series (tokens vs turn).
