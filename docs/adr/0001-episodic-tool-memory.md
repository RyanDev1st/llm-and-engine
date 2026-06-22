# ADR 0001 — Episodic "how-to-operate" memory (the system learns, the model stays frozen)

Status: Accepted (built, flag-gated default-off) — 2026-06-22

## Context
The product model (Gemma 4 E4B v4 adapter) is FROZEN. The captured serve transcript showed a
recurring quirk on OOD tools: the model fires a tool with no args (`<tool>scale_recipe>`), reads the
error, then fixes it (`<tool>scale_recipe from_servings=12 to_servings=30>`). It always self-heals to
a grounded answer, but pays an extra step every time — and pays it AGAIN on the next similar request,
because nothing persists what it learned. The user asked: can the system self-learn as conversations
go, persistently across runs and machines, without retraining?

Retraining (continual/online fine-tuning) was rejected: it needs a GPU loop, risks catastrophic
forgetting of the 98.6% routing, and "persist across machines" would mean shipping adapter checkpoints
— heavy and fragile for a frozen-GGUF-on-a-4060 product.

## Decision
Add a FOURTH memory tier — **episodic, global, file-backed** — alongside the existing
ephemeral / session / per-user-profile tiers (`backend/memory/episodic.py`):

- **The signal is free.** The serve loop already produces a correction whenever a tool errors then
  succeeds in the same turn. `observe(user_message, result, plugin_context)` harvests that as an
  episode: `{tool, trigger (the request), lesson (the call that WORKED), hits}`.
- **Recall is cheap + domain-general.** `episodic_block(user_message, plugin_context)` finds the
  closest past episode by LEXICAL similarity (Jaccard over content tokens — no model, no embeddings)
  and injects a one-line HINT, but ONLY for a tool that is in the LIVE manifest. It keys on the
  request text + manifest, not on chess — so it works for any plugin/domain.
- **Global, not per-user.** A lesson is about operating a TOOL, so it benefits every user and every
  machine. One JSON file under `CHESS_MEMORY_DIR`; sync that dir → cross-machine learning, no infra.
- **Wired where profile memory already is** (`web_app._context_block` injects, `_run` harvests) — the
  core loop (`respond`/`_finalize`) is untouched, so there is zero core-loop risk.
- **Flag-gated** `CHESS_EPISODIC=1`, default OFF → current serve behavior is byte-identical until
  proven on the GPU flight.

## Consequences
- **Self-learning without weight change** ("the *system* learns, not the model"): the right kind for a
  frozen product, and it composes with the verb-coercion / grounding / coverage layers already shipped.
- **Poisoning is the central risk** (Cursor removed ungated auto-memory for this reason). Mitigated by a
  strict gate: harvest ONLY a turn that reached a grounded answer (reply present) via a same-tool
  error→fix; reject PII/board-state (reuse `store._REJECT`); one lesson per tool (newest refreshes);
  bounded store (least-used evicted). A turn that ended in error is never a lesson.
- **Train/serve skew**: the model never trained with a RECALLED block. Mitigated by reusing the
  injected-context idiom it already tolerates (profile + live-board), one short line, framed as a hint.
- **Proof is a GPU-flight item**: a recurrence test (same request twice — does occurrence #2 go
  one-shot, `recovered`↓ / `first_ok`↑ in the completion eval) measures the real frozen-model effect.
  The mechanism is unit-tested on CPU (`test_episodic.py`); the magnitude awaits the flight.

## Alternatives rejected
- Continual/online fine-tuning — see Context (heavy, forgetting risk, checkpoint-shipping).
- Per-user episodic store — a tool-usage lesson isn't user-specific; global shares the benefit.
- Embedding/vector recall — adds a dependency + latency for a bounded store that lexical Jaccard
  handles; revisit only if the store grows large or cross-lingual recall is needed.
- Reach for it via a global regex arg-filler in the deterministic layer — rejected per the
  `deterministic-routing-restraint` lesson; episodic memory is data-driven and per-tool, not regex.
